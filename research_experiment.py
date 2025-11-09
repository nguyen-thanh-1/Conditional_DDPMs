import os
import math
import random
import time
import json
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# --- NEW: Imports for this script ---
from accelerate import Accelerator
from fvcore.nn import FlopCountAnalysis
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
#from diffusers.utils import get_polynomial_decay_schedule_with_warmup

# --- Base Imports (from old file) ---
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from transformers import get_polynomial_decay_schedule_with_warmup
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
from safetensors.torch import load_file
import warnings

warnings.filterwarnings("ignore", message=".triton.*")
# os.environ["XFORMERS_DISABLE_FLASH_ATTENTION"] = "1" # Accelerate có thể tự xử lý
# os.environ["XFORMERS_DISABLE_TRITON_WARNINGS"] = "1"

# ---------------------------
# 1. NEW: Argument Parser
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser()
    
    # --- Paths and IO ---
    parser.add_argument("--DATA_ROOT", type=str, default="1_percent_dataset", help="Path to your dataset")
    parser.add_argument("--MODEL_ID", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model ID")
    parser.add_argument("--OUTPUT_DIR", type=str, required=True, help="Where to save results for this *specific* run")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--BATCH_SIZE", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--NUM_EPOCHS", type=int, default=20)
    parser.add_argument("--LR", type=float, default=1e-5, help="Base learning rate (for UNet)")
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--IMAGE_SIZE", type=int, default=128)
    parser.add_argument("--SAVE_EVERY", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--NUM_WORKERS", type=int, default=4)
    
    # --- NEW: Scheduler Config ---
    parser.add_argument("--SCHEDULER_TYPE", type=str, default="none", choices=["none", "cosine", "onecycle", "linear_warmup"])
    parser.add_argument("--LR_WARMUP_STEPS", type=int, default=500, help="Warmup steps for linear_warmup scheduler")

    # --- CLIP Fine-tuning Config ---
    parser.add_argument("--N_TEXT_UNFREEZE", type=int, default=2, help="Unfreeze last N layers of CLIP")
    parser.add_argument("--PHASE1_EPOCHS", type=int, default=2, help="Epochs to train UNet only before unfreezing CLIP")
    parser.add_argument("--TEXT_LR_RATIO", type=float, default=0.2, help="CLIP learning rate = LR * TEXT_LR_RATIO")
    # --- THÊM DÒNG NÀY ---
    parser.add_argument("--NUM_INFERENCE_STEPS", type=int, default=50, help="Number of denoising steps for evaluation sampling")
    # --- KẾT THÚC ---
    # --- THÊM 2 DÒNG NÀY ---
    parser.add_argument("--override_beta_schedule", type=str, default=None, help="Override beta_schedule (e.g., 'cosine')")
    parser.add_argument("--override_num_train_timesteps", type=int, default=None, help="Override num_train_timesteps (e.g., 800)")
    # --- KẾT THÚC ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint folder to resume training from")
    return parser.parse_args()

# ---------------------------
# 2. Dataset (Copied from old file)
# ---------------------------
class TextSketchDataset(Dataset):
    def __init__(self, split_dir, tokenizer, image_size=128, max_length=65):
        self.image_files = sorted(glob(os.path.join(split_dir, "images", "*.*")))
        self.text_dir = os.path.join(split_dir, "texts")
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]

        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        txt_path = os.path.join(self.text_dir, name + ".txt")
        caption = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.readline().strip()

        tokenized = self.tokenizer(caption, padding="max_length", truncation=True,
                                   max_length=self.max_length, return_tensors="pt")
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        return {
            "pixel_values": img_t,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,
            "name": name
        }

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    captions = [b["caption"] for b in batch]
    names = [b["name"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "captions": captions,
        "names": names
    }

# ---------------------------
# 3. Model Helpers (Copied & Refactored)
# ---------------------------
def set_text_encoder_trainable(text_encoder, n_unfreeze=1):
    for _, p in text_encoder.named_parameters():
        p.requires_grad_(False)
    
    if n_unfreeze <= 0:
        return

    try:
        layers = text_encoder.text_model.encoder.layers
    except AttributeError:
        layers = text_encoder.encoder.layers
    
    total = len(layers)
    for i in range(total - n_unfreeze, total):
        for p in layers[i].parameters():
            p.requires_grad_(True)
            
    for name, p in text_encoder.named_parameters():
        if "final_layer_norm" in name or "proj" in name or "text_projection" in name:
            p.requires_grad_(True)

def make_optimizer(unet, text_encoder, base_lr, text_lr_ratio):
    """UNet lr full, CLIP lr smaller"""
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    text_params = [p for p in text_encoder.parameters() if p.requires_grad]
    
    groups = [{"params": unet_params, "lr": base_lr}]
    
    n_clip_params = sum(p.numel() for p in text_params)
    
    if text_params:
        groups.append({"params": text_params, "lr": base_lr * text_lr_ratio})
        
    optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
    return optimizer, n_clip_params

# NEW: Scheduler creation
def make_scheduler(optimizer, scheduler_type, num_train_steps, warmup_steps=0):
    if scheduler_type == "none":
        return None
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_train_steps)
    elif scheduler_type == "onecycle":
        # OneCycleLR needs max_lr for each param group
        max_lrs = [g['lr'] for g in optimizer.param_groups]
        return OneCycleLR(optimizer, max_lr=max_lrs, total_steps=num_train_steps)
    elif scheduler_type == "linear_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps,
            lr_end=1e-7, # small value
            power=1.0 # linear
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

# ---------------------------
# 4. Metrics Class (Copied from old file)
# ---------------------------
class Metrics:
    def __init__(self, device):
        # device is the main accelerator device (e.g., 'cuda:0')
        # We force metrics to CPU to save VRAM
        self.cpu_device = torch.device("cpu")
        
        print("Loading metric models on CPU...")
        self.fid_metric = FrechetInceptionDistance(feature=64).to(self.cpu_device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.cpu_device)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.cpu_device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def lpips(self, img1, img2):
        # imgs are on GPU, move to CPU
        img1_cpu = img1.detach().to(self.cpu_device)
        img2_cpu = img2.detach().to(self.cpu_device)
        
        if img1_cpu.max() <= 1.0: img1_cpu = img1_cpu * 2 - 1
        if img2_cpu.max() <= 1.0: img2_cpu = img2_cpu * 2 - 1
            
        with torch.no_grad():
            return self.lpips_loss(img1_cpu, img2_cpu).mean().item()

    def clip_text_image_score(self, images, texts):
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        prepared_images = [to_pil(i.cpu()) if isinstance(i, torch.Tensor) else i for i in images]
        
        inputs = self.clip_processor(text=texts, images=prepared_images, return_tensors="pt", padding=True).to(self.cpu_device)
        
        with torch.no_grad():
            out = self.clip(**inputs)
            img, txt = out.image_embeds, out.text_embeds
            img = img / img.norm(p=2, dim=-1, keepdim=True)
            txt = txt / txt.norm(p=2, dim=-1, keepdim=True)
            cos = (img * txt).sum(dim=-1)
            return cos.mean().item()

    def fid(self, gen_images, real_images):
        gen_images_cpu = (gen_images * 255).byte().detach().to(self.cpu_device)
        real_images_cpu = (real_images * 255).byte().detach().to(self.cpu_device)
        
        self.fid_metric.reset()
        self.fid_metric.update(real_images_cpu, real=True)
        self.fid_metric.update(gen_images_cpu, real=False)
        fid_score = self.fid_metric.compute().item()
        return fid_score

# ---------------------------
# 5. NEW: Performance & Logging Helpers
# ---------------------------
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def log_performance_metrics(vae, unet, text_encoder, image_size, max_length, output_dir):
    """Logs FLOPs and Trainable Parameters to a file."""
    print("Calculating FLOPs and Parameters...")
    
    # --- Dummy Inputs ---
    # We send dummy inputs to CPU to avoid potential DDP/FSDP issues with fvcore
    cpu_device = torch.device("cpu")
    vae = vae.to(cpu_device)
    unet = unet.to(cpu_device)
    text_encoder = text_encoder.to(cpu_device)

    dummy_image = torch.randn((1, 3, image_size, image_size), device=cpu_device)
    dummy_latents = vae.encode(dummy_image * 2.0 - 1.0).latent_dist.sample()
    dummy_t = torch.tensor([1], device=cpu_device)
    dummy_ids = torch.randint(0, 1000, (1, max_length), device=cpu_device)
    dummy_hidden_states = text_encoder(dummy_ids).last_hidden_state

    # --- FLOPs ---
    flops_vae_enc = FlopCountAnalysis(vae.encoder, dummy_image)
    flops_vae_dec = FlopCountAnalysis(vae.decoder, dummy_latents)
    flops_text = FlopCountAnalysis(text_encoder, dummy_ids)
    flops_unet = FlopCountAnalysis(unet, (dummy_latents, dummy_t, dummy_hidden_states))
    
    total_gflops_step = (flops_text.total() + flops_unet.total()) / 1e9
    
    # --- Parameters ---
    unet_trainable = count_trainable_params(unet)
    clip_base_trainable = count_trainable_params(text_encoder) # (will be 0 if not unfrozen yet)

    # --- Save Report ---
    report = {
        "GFLOPs_per_train_step (UNet + TextEncoder)": f"{total_gflops_step:.2f}",
        "GFLOPs_UNet_step": f"{flops_unet.total() / 1e9:.2f}",
        "GFLOPs_TextEncoder_step": f"{flops_text.total() / 1e9:.2f}",
        "GFLOPs_VAE_Encoder": f"{flops_vae_enc.total() / 1e9:.2f}",
        "GFLOPs_VAE_Decoder": f"{flops_vae_dec.total() / 1e9:.2f}",
        "params_UNet_trainable": unet_trainable,
        "params_CLIP_initial_trainable": clip_base_trainable,
    }
    
    report_path = os.path.join(output_dir, "performance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Performance report saved to {report_path}")
    
    # Move models back just in case (though accelerator.prepare will handle device)
    vae.to(torch.device("cpu")) # Keep VAE on CPU
    unet.to(torch.device("cpu"))
    text_encoder.to(torch.device("cpu"))
    
    return report # return the report dict

def save_config(args, path):
    """Saves all argparse args to a JSON file."""
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=4)

def init_metrics_csv(path, columns):
    """Creates a new CSV file with headers if it doesn't exist."""
    if not os.path.exists(path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)

def log_metrics_to_csv(path, metrics_dict):
    """Appends a new row of metrics to the CSV."""
    if not os.path.exists(path):
        # Create file with headers from dict keys if it's the first time
        init_metrics_csv(path, list(metrics_dict.keys()))
        
    df = pd.DataFrame([metrics_dict])
    df.to_csv(path, mode="a", header=False, index=False)

# ---------------------------
# 6. Plotting (Copied & Refactored)
# ---------------------------
def plot_all_curves(output_dir):
    """Reads the final CSV and plots all metrics."""
    metrics_path = os.path.join(output_dir, "training_log.csv")
    if not os.path.exists(metrics_path):
        print("metrics_log.csv not found. Skipping plotting.")
        return

    df = pd.read_csv(metrics_path)

    # --- Plot 1: Loss vs. Epoch ---
    # (Đã cập nhật: thêm plt.ylim)
    plt.figure(figsize=(10,5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.ylim(0, 1) # <-- THAY ĐỔI: Cố định trục Y để mượt hơn
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plot_epoch_loss.png"))
    plt.close()

    # --- Plot 2: Quality vs. Epoch (Grid) ---
    # (Đã cập nhật: Tách thành 3 biểu đồ con)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Quality Metrics vs. Epoch', fontsize=16)

    # LPIPS vs Epoch
    ax1.plot(df["epoch"], df["LPIPS"], label="LPIPS (lower is better)", color='blue')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("LPIPS")
    ax1.legend()
    ax1.grid(True)

    # FID vs Epoch
    ax2.plot(df["epoch"], df["FID"], label="FID (lower is better)", color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("FID")
    ax2.legend()
    ax2.grid(True)

    # CLIP-score vs Epoch
    ax3.plot(df["epoch"], df["CLIP_score"], label="CLIP-score (higher is better)", color='purple')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("CLIP-score")
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Dành không gian cho tiêu đề chung
    plt.savefig(os.path.join(output_dir, "plot_epoch_quality_grid.png"))
    plt.close()

    # --- Plot 3: Loss vs. Time ---
    # (Đã cập nhật: thêm plt.ylim)
    plt.figure(figsize=(10,5))
    plt.plot(df["elapsed_time_min"], df["train_loss"], label="Train Loss")
    plt.plot(df["elapsed_time_min"], df["val_loss"], label="Val Loss")
    plt.xlabel("Training Time (minutes)")
    plt.ylabel("Loss")
    plt.title("Loss vs. Training Time")
    plt.ylim(0, 1) # <-- THAY ĐỔI: Cố định trục Y để mượt hơn
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plot_time_loss.png"))
    plt.close()
    
    # --- Plot 4: Quality vs. Time (Grid) ---
    # (Đã cập nhật: Tách thành 3 biểu đồ con)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Quality Metrics vs. Training Time', fontsize=16)

    # LPIPS vs Time
    ax1.plot(df["elapsed_time_min"], df["LPIPS"], label="LPIPS (lower is better)", color='blue')
    ax1.set_xlabel("Training Time (minutes)")
    ax1.set_ylabel("LPIPS")
    ax1.legend()
    ax1.grid(True)

    # FID vs Time
    ax2.plot(df["elapsed_time_min"], df["FID"], label="FID (lower is better)", color='green')
    ax2.set_xlabel("Training Time (minutes)")
    ax2.set_ylabel("FID")
    ax2.legend()
    ax2.grid(True)

    # CLIP-score vs Time
    ax3.plot(df["elapsed_time_min"], df["CLIP_score"], label="CLIP-score (higher is better)", color='purple')
    ax3.set_xlabel("Training Time (minutes)")
    ax3.set_ylabel("CLIP-score")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Dành không gian cho tiêu đề chung
    plt.savefig(os.path.join(output_dir, "plot_time_quality_grid.png"))
    plt.close()

    print(f"All plots saved to: {output_dir}")


# ---------------------------
# 7. Validation (Refactored for Accelerate)
# ---------------------------
@torch.no_grad()
def validate_one_epoch(vae, unet, text_encoder, tokenizer, noise_scheduler, val_loader, accelerator):
    unet.eval()
    text_encoder.eval()
    
    total, n = 0.0, 0
    # REFACTORED: No tqdm, will be wrapped by accelerator
    for batch in val_loader:
        # Note: val_loader is already 'prepared' by accelerate, no .to(device) needed
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        
        # We need to manually move VAE inputs if VAE is not part of 'prepare'
        # Safest way: keep VAE on CPU and move data
        latents = encode_images_to_latents(vae, pixel_values.to(vae.device))
        latents = latents.to(accelerator.device) # Move latents to main device
        
        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=latents.device).long()
        noisy = noise_scheduler.add_noise(latents, noise, t)
        
        # Models are already on the correct device
        hidden = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        pred = unet(noisy, t, hidden).sample
        
        loss = F.mse_loss(pred, noise)
        
        total += loss.item() * latents.size(0)
        n += latents.size(0)
        
    return total / max(1, n)

@torch.no_grad()
def generate_images_for_eval(unet, vae, text_encoder, noise_scheduler, accelerator, batch, num_inference_steps):
    """Generates images from a batch of text prompts for evaluation."""
    unet.eval()
    text_encoder.eval()

    # Dữ liệu đã ở trên device của accelerator
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    real_images = batch["pixel_values"] # Dùng để lấy shape
    
    # 1. Lấy Text Embeddings
    encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    # 2. Lấy thông số VAE (đang ở CPU)
    vae_scaling_factor = vae.config.scaling_factor
    
    # Mã hóa ảnh thật (trên CPU) để lấy shape của latent
    latents_shape = encode_images_to_latents(vae, real_images).shape
    
    # 3. Tạo latents ban đầu trên device (GPU)
    latents = torch.randn(latents_shape, device=accelerator.device, dtype=encoder_hidden_states.dtype)
    
    # 4. Đặt timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    timesteps = noise_scheduler.timesteps

    # 5. Vòng lặp lọc nhiễu
    for t in timesteps:
        noise_pred = unet(latents, t, encoder_hidden_states).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # 6. Giải mã latents (chuyển về CPU cho VAE)
    latents_cpu = latents.to(vae.device)
    latents_cpu = latents_cpu / vae_scaling_factor
    imgs_cpu = vae.decode(latents_cpu).sample
    
    imgs_cpu = (imgs_cpu / 2 + 0.5).clamp(0, 1)
    
    # Trả về tensor trên CPU (vì các model metric cũng ở CPU)
    return imgs_cpu

@torch.no_grad()
def encode_images_to_latents(vae, images):
    # VAE is on CPU, move images to CPU
    images_cpu = images.to(vae.device)
    images_cpu = 2.0 * images_cpu - 1.0
    latents = vae.encode(images_cpu).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


# ---------------------------
# 8. Main Training Function
# ---------------------------
def main():
    args = get_args()
    
    # --- 1. Setup Accelerator ---
    # NEW: Accelerator handles device placement, multi-GPU, fp16
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        log_with="tensorboard", # You can change to "wandb"
        project_dir=args.OUTPUT_DIR
    )
    accelerator.init_trackers("research_project") # Tboard dir will be OUTPUT_DIR/research_project

    # Set seed for reproducibility
    random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    
    # Create output dir
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    
    # --- 2. Load Components ---
    # REFACTORED: Load models to CPU first. Accelerator will handle moving.
    # VAE stays on CPU to save VRAM
    vae = AutoencoderKL.from_pretrained(args.MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.MODEL_ID, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(args.MODEL_ID, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.MODEL_ID, subfolder="tokenizer")
    # 1. Tải config của scheduler gốc về dưới dạng dictionary
    scheduler_config = DDPMScheduler.load_config(args.MODEL_ID, subfolder="scheduler")

    # 2. Kiểm tra xem người dùng có muốn ghi đè không
    if args.override_beta_schedule is not None:
        accelerator.print(f"WARNING: Overriding beta_schedule. Default was '{scheduler_config['beta_schedule']}', using '{args.override_beta_schedule}' instead.")
        scheduler_config['beta_schedule'] = args.override_beta_schedule

    if args.override_num_train_timesteps is not None:
        accelerator.print(f"WARNING: Overriding num_train_timesteps. Default was '{scheduler_config['num_train_timesteps']}', using '{args.override_num_train_timesteps}' instead.")
        scheduler_config['num_train_timesteps'] = args.override_num_train_timesteps

    # 3. Tạo noise_scheduler từ config (đã được cập nhật hoặc vẫn là gốc)
    noise_scheduler = DDPMScheduler.from_config(scheduler_config)

    # 4. Lưu lại các giá trị CUỐI CÙNG (dù là gốc hay đã override) để ghi log
    args.num_train_timesteps = noise_scheduler.config.num_train_timesteps
    args.beta_schedule = noise_scheduler.config.beta_schedule
    # --- 3. Save Config ---
    # NEW: Save all args to a file for perfect reproducibility
    if accelerator.is_main_process:
        save_config(args, os.path.join(args.OUTPUT_DIR, "run_config.json"))

    vae.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    
    # --- 4. Log Performance (FLOPs, Params) ---
    # NEW: Run this once on the main process before we unfreeze CLIP
    if accelerator.is_main_process:
        # Create a deep copy for analysis to avoid state issues
        from copy import deepcopy
        perf_report = log_performance_metrics(
            deepcopy(vae), deepcopy(unet), deepcopy(text_encoder), 
            args.IMAGE_SIZE, tokenizer.model_max_length, args.OUTPUT_DIR
        )

    # --- 5. Setup Dataset & Loaders ---
    train_ds = TextSketchDataset(os.path.join(args.DATA_ROOT, "train"), tokenizer, image_size=args.IMAGE_SIZE)
    val_ds = TextSketchDataset(os.path.join(args.DATA_ROOT, "val"), tokenizer, image_size=args.IMAGE_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=args.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=args.NUM_WORKERS)
    
    # --- 6. Setup Optimizer, Scheduler, Phase 1 ---
    # Phase 1: Only UNet is trainable
    set_text_encoder_trainable(text_encoder, n_unfreeze=0)
    optimizer, _ = make_optimizer(unet, text_encoder, args.LR, args.TEXT_LR_RATIO)
    
    # Calculate total steps
    num_train_steps = len(train_loader) * args.NUM_EPOCHS
    
    scheduler = make_scheduler(optimizer, args.SCHEDULER_TYPE, num_train_steps, args.LR_WARMUP_STEPS)

    # --- 7. ACCELERATE PREPARE ---
    # REFACTORED: This is the core of Accelerate
    unet, text_encoder, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_loader, val_loader, scheduler
    )
    
    # VAE is NOT prepared, keep it on CPU
    vae = vae.to(torch.device("cpu"))
    
    # --- 8. Init Metrics & Logging ---

    # --- NEW: Check for Resuming ---
    start_epoch = 0
    global_step_offset = 0

    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        # --- NEW: Manual Loading (Forced to CPU) ---
        # 1. Load UNet (model 0)
        unet_path = os.path.join(args.resume_from_checkpoint, "model.safetensors")
        if os.path.exists(unet_path):
            # THAY ĐỔI: Tải lên "cpu" để tránh lỗi
            unet_state_dict = load_file(unet_path, device="cpu") 
            accelerator.unwrap_model(unet).load_state_dict(unet_state_dict)
            accelerator.print("Loaded UNet state.")
            del unet_state_dict # Giải phóng bộ nhớ
        else:
            accelerator.print(f"WARNING: No model.safetensors found in {args.resume_from_checkpoint}")

        # 2. Load Text Encoder (model 1)
        text_encoder_path = os.path.join(args.resume_from_checkpoint, "model_1.safetensors")
        if os.path.exists(text_encoder_path):
            # THAY ĐỔI: Tải lên "cpu" để tránh lỗi
            text_encoder_state_dict = load_file(text_encoder_path, device="cpu")
            accelerator.unwrap_model(text_encoder).load_state_dict(text_encoder_state_dict)
            accelerator.print("Loaded Text Encoder state.")
            del text_encoder_state_dict # Giải phóng bộ nhớ
        elif args.N_TEXT_UNFREEZE > 0:
             accelerator.print(f"WARNING: N_TEXT_UNFREEZE > 0 but no model_1.safetensors found.")
        
        # 3. Load Optimizer
        optimizer_path = os.path.join(args.resume_from_checkpoint, "optimizer.bin")
        if os.path.exists(optimizer_path):
            # THAY ĐỔI: Map_location về "cpu"
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            accelerator.print("Loaded Optimizer state.")
        else:
            accelerator.print(f"WARNING: No optimizer.bin found in {args.resume_from_checkpoint}")

        # 4. Load Scaler
        scaler_path = os.path.join(args.resume_from_checkpoint, "scaler.pt")
        if os.path.exists(scaler_path) and accelerator.scaler is not None:
            # THAY ĐỔI: Map_location về "cpu"
            accelerator.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
            accelerator.print("Loaded Scaler state.")
        
        # 5. Load Scheduler (if it exists)
        scheduler_path = os.path.join(args.resume_from_checkpoint, "scheduler.bin")
        if os.path.exists(scheduler_path) and scheduler is not None:
             # THAY ĐỔI: Map_location về "cpu"
             scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
             accelerator.print("Loaded Scheduler state.")
             
        # --- END Manual Loading ---

        # Tự động suy ra epoch bắt đầu từ tên thư mục
        try:
            completed_epoch = int(args.resume_from_checkpoint.rstrip('/').split('_')[-1])
            start_epoch = completed_epoch # Bắt đầu từ epoch tiếp theo (vì epoch 8 đã xong)
            accelerator.print(f"Resuming from epoch {start_epoch}")
            global_step_offset = start_epoch * len(train_loader)
        except Exception as e:
            accelerator.print(f"WARNING: Could not infer epoch from checkpoint name. {e}")
            
    # --- END NEW ---

    if accelerator.is_main_process:
        metric_eval = Metrics(accelerator.device) # Pass main device
        metrics_csv_path = os.path.join(args.OUTPUT_DIR, "training_log.csv")
        
        # --- NEW: Chỉ tạo file CSV mới nếu không phải là resume ---
        if not args.resume_from_checkpoint:
            init_metrics_csv(metrics_csv_path, [
                "epoch", "elapsed_time_min", "train_loss", "val_loss", 
                "LPIPS", "FID", "CLIP_score", "current_lr", "trainable_clip_params"
            ])
    
    global_step = global_step_offset # <-- THAY ĐỔI: Dùng offset
    start_time = time.time()
    
    # --- 9. Training Loop ---
    accelerator.print(f"--- Starting Training from epoch {start_epoch} for {args.NUM_EPOCHS} Epochs ---")
    
    # --- THAY ĐỔI: Sửa vòng lặp for ---
    for epoch in range(start_epoch, args.NUM_EPOCHS):
    # --- KẾT THÚC THAY ĐỔI ---
        
        # --- Check for Phase 2: Unfreeze CLIP ---
        # NEW: Handle unfreezing and re-creating optimizer mid-training
        if epoch == args.PHASE1_EPOCHS and args.N_TEXT_UNFREEZE > 0:
            accelerator.print(f">>> Phase 2: Unfreezing last {args.N_TEXT_UNFREEZE} CLIP layers.")
            
            # 1. Unwrap models to modify them
            unet_unwrapped = accelerator.unwrap_model(unet)
            text_encoder_unwrapped = accelerator.unwrap_model(text_encoder)
            
            # 2. Apply change
            set_text_encoder_trainable(text_encoder_unwrapped, n_unfreeze=args.N_TEXT_UNFREEZE)
            
            # 3. Re-create optimizer & scheduler
            new_optimizer, n_clip_params = make_optimizer(unet_unwrapped, text_encoder_unwrapped, args.LR, args.TEXT_LR_RATIO)
            new_scheduler = make_scheduler(new_optimizer, args.SCHEDULER_TYPE, num_train_steps, args.LR_WARMUP_STEPS)
            
            accelerator.print(f"New Trainable CLIP Params: {n_clip_params}")

            # 4. Re-prepare them
            optimizer, scheduler = accelerator.prepare(new_optimizer, new_scheduler)
            
            # Note: We must also update the perf report on main process
            if accelerator.is_main_process:
                perf_report["params_CLIP_unfrozen_trainable"] = n_clip_params
                with open(os.path.join(args.OUTPUT_DIR, "performance_report.json"), "w") as f:
                    json.dump(perf_report, f, indent=4)
        
        # --- Train Step ---
        unet.train()
        text_encoder.train() # Train if unfrozen
        
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", disable=not accelerator.is_local_main_process)
        
        for batch in pbar:
            with accelerator.accumulate(unet): # Handles gradient accumulation
                # Data is already on the correct device
                pixel_values = batch["pixel_values"]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                # REFACTORED: VAE encoding on-the-fly (on CPU)
                latents = encode_images_to_latents(vae, pixel_values)
                latents = latents.to(accelerator.device) # Move to GPU
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps,
                                          (latents.size(0),), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # REFACTORED: No with torch.no_grad()
                encoder_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                encoder_hidden_states = encoder_outputs.last_hidden_state
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                
                optimizer.step()
                if scheduler and args.SCHEDULER_TYPE in ["onecycle", "linear_warmup"]:
                    scheduler.step() # Step per batch
                optimizer.zero_grad()
                
                running_loss += loss.item()
                global_step += 1
                
                if accelerator.is_local_main_process:
                    pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.6f}"})

        # End of train epoch
        avg_train_loss = running_loss / len(train_loader)
        
        if scheduler and args.SCHEDULER_TYPE in ["cosine"]:
            scheduler.step() # Step per epoch
            
        # --- Validation & Metrics ---
        if accelerator.is_main_process:
            accelerator.print("Running validation...")
            val_loss = validate_one_epoch(vae, unet, text_encoder, tokenizer, noise_scheduler, val_loader, accelerator)
            
            # --- Sample Evaluation (LPIPS, FID, CLIP) ---
            # (Using a small, fixed subset of val_loader for speed)
            # This logic is simplified from your file. You can restore your K-sample logic here.
            
            # Lấy 1 batch cố định từ val
            try:
                eval_batch = next(iter(val_loader))
                real_images = eval_batch["pixel_values"]
                captions = eval_batch["captions"]
                
                # Generate images (simplified)
                # (You should replace this with your full 'generate_samples_for_eval' logic)
                #gen_images = torch.randn_like(real_images) # Placeholder
                gen_images = generate_images_for_eval(
                    accelerator.unwrap_model(unet), # Cần unwrap model
                    vae, 
                    accelerator.unwrap_model(text_encoder), 
                    noise_scheduler, 
                    accelerator, 
                    eval_batch, 
                    args.NUM_INFERENCE_STEPS # <-- Dùng tham số mới
                )
                
                # --- Tính toán mọi chỉ số ---
                lpips_score = metric_eval.lpips(gen_images.to(accelerator.device), real_images.to(accelerator.device))
                fid_score = metric_eval.fid(gen_images, real_images)
                clip_score = metric_eval.clip_text_image_score(list(gen_images), captions)
            
            except Exception as e:
                accelerator.print(f"Warning: Metric evaluation failed. {e}")
                lpips_score, fid_score, clip_score = None, None, None

            # --- Log everything ---
            elapsed_time_min = (time.time() - start_time) / 60.0
            current_lr = optimizer.param_groups[0]['lr']
            trainable_clip_params = perf_report.get("params_CLIP_unfrozen_trainable", 0)
            
            metrics_dict = {
                "epoch": epoch + 1,
                "elapsed_time_min": elapsed_time_min,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "LPIPS": lpips_score,
                "FID": fid_score,
                "CLIP_score": clip_score,
                "current_lr": current_lr,
                "trainable_clip_params": trainable_clip_params
            }
            
            log_metrics_to_csv(metrics_csv_path, metrics_dict)
            accelerator.log(metrics_dict, step=epoch+1) # Log to Tensorboard
            
            accelerator.print(f"Epoch {epoch+1} Metrics: {metrics_dict}")

            # --- Save Model ---
            if (epoch + 1) % args.SAVE_EVERY == 0:
                accelerator.save_state(os.path.join(args.OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}"))
    
    # --- 10. End of Training ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print("Training finished. Saving final model and plots.")
        
        # Save final state
        accelerator.save_state(os.path.join(args.OUTPUT_DIR, "final_model"))
        
        # Unwrap and save pretrained (optional, for easy sharing)
        unet_unwrapped = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(os.path.join(args.OUTPUT_DIR, "final_unet_pretrained"))
        
        if args.N_TEXT_UNFREEZE > 0:
            text_encoder_unwrapped = accelerator.unwrap_model(text_encoder)
            text_encoder_unwrapped.save_pretrained(os.path.join(args.OUTPUT_DIR, "final_text_encoder_pretrained"))
            
        # --- Generate Final Plots ---
        plot_all_curves(args.OUTPUT_DIR)
        
    accelerator.end_training()


if __name__ == "__main__":
    main()