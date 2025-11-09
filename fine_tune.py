import os
import math
import random
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from torchvision import transforms, utils
from torchmetrics.image.fid import FrechetInceptionDistance
import pandas as pd
from safetensors.torch import load_file

import lpips  # pip install lpips
import warnings, os

warnings.filterwarnings("ignore", message=".triton.*")
os.environ["XFORMERS_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["XFORMERS_DISABLE_TRITON_WARNINGS"] = "1"

# ---------------------------
# Configuration
# ---------------------------
DATA_ROOT = "dataset"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "sd_text2sketch_finetuned-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-5
SEED = 42
IMAGE_SIZE = 128
SAVE_EVERY = 4
NUM_WORKERS = 4

TRAIN_SUBSET_FRAC = 1.0  # set <1.0 for debug

# NEW CONFIGS
N_TEXT_UNFREEZE = 2      # unfreeze last N layers of CLIP
PHASE1_EPOCHS = 2        # epochs to train UNet only before unfreezing CLIP
TEXT_LR_RATIO = 0.2      # CLIP learning rate = LR * TEXT_LR_RATIO


# ---------------------------
# Utilities / Dataset
# ---------------------------
class TextSketchDataset(Dataset):
    def __init__(self, split_dir, tokenizer, image_size=128, max_length=77):
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
# 🔧 NEW: Partial CLIP fine-tuning helpers
# ---------------------------
def set_text_encoder_trainable(text_encoder, n_unfreeze=1):
    """Unfreeze last n transformer blocks + final layer norm/projection"""
    for _, p in text_encoder.named_parameters():
        p.requires_grad_(False)
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


def make_optimizer(unet, text_encoder, base_lr):
    """UNet lr full, CLIP lr smaller"""
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    text_params = [p for p in text_encoder.parameters() if p.requires_grad]
    groups = [{"params": unet_params, "lr": base_lr}]
    if text_params:
        groups.append({"params": text_params, "lr": base_lr * TEXT_LR_RATIO})
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)


# ---------------------------
# Load pretrained components
# ---------------------------
def load_components(model_id=MODEL_ID, device=DEVICE, n_text_unfreeze=0):
    print("Loading components from", model_id)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    unet.enable_gradient_checkpointing()
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if n_text_unfreeze > 0:
        set_text_encoder_trainable(text_encoder, n_unfreeze=n_text_unfreeze)

    return vae, unet, text_encoder, tokenizer, noise_scheduler


# ---------------------------
# Helper: encode images to latents
# ---------------------------
@torch.no_grad()
def encode_images_to_latents(vae, images):
    images = 2.0 * images - 1.0
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


# ---------------------------
# Metrics
# ---------------------------
class Metrics:
    def __init__(self, device):
        self.device = device  # 'cuda'
        self.cpu_device = torch.device("cpu")
        
        print("Loading metric models on CPU to save VRAM...")
        # Load all metric models to CPU
        self.fid_metric = FrechetInceptionDistance(feature=64).to(self.cpu_device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.cpu_device)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.cpu_device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def lpips(self, img1, img2):
        # img1, img2 đang ở trên GPU, chuyển chúng về CPU
        img1_cpu = img1.detach().to(self.cpu_device)
        img2_cpu = img2.detach().to(self.cpu_device)
        
        if img1_cpu.max() <= 1.0:
            img1_cpu = img1_cpu * 2 - 1
        if img2_cpu.max() <= 1.0:
            img2_cpu = img2_cpu * 2 - 1
            
        with torch.no_grad():
            return self.lpips_loss(img1_cpu, img2_cpu).mean().item()

    def clip_text_image_score(self, images, texts, device):
        # images là list các tensor/PIL (đã ở trên CPU), texts là list string
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        prepared_images = [to_pil(i.cpu()) if isinstance(i, torch.Tensor) else i for i in images]
        
        # Chạy processor và model trên CPU
        inputs = self.clip_processor(text=texts, images=prepared_images, return_tensors="pt", padding=True).to(self.cpu_device)
        
        with torch.no_grad():
            out = self.clip(**inputs) # Model chạy trên CPU
            img, txt = out.image_embeds, out.text_embeds
            img = img / img.norm(p=2, dim=-1, keepdim=True)
            txt = txt / txt.norm(p=2, dim=-1, keepdim=True)
            cos = (img * txt).sum(dim=-1)
            return cos.mean().item()

    def fid(self, gen_images, real_images):
        # gen_images, real_images đang ở trên GPU, chuyển về CPU
        gen_images_cpu = (gen_images * 255).byte().detach().to(self.cpu_device)
        real_images_cpu = (real_images * 255).byte().detach().to(self.cpu_device)
        
        self.fid_metric.reset()
        self.fid_metric.update(real_images_cpu, real=True)
        self.fid_metric.update(gen_images_cpu, real=False)
        fid_score = self.fid_metric.compute().item()
        return fid_score
# ---------------------------
# Training / Validation
# ---------------------------
def train_loop(vae, unet, text_encoder, tokenizer, noise_scheduler,
               train_loader, val_loader, device, config):

    #optimizer = make_optimizer(unet, text_encoder, config["lr"])
    metric = Metrics(device)

    best_val_loss = float("inf")
    os.makedirs(config["out_dir"], exist_ok=True)
    
    start_epoch = 0
    optimizer_state = None # Placeholder
    
    # --- THAY BẰNG LOGIC NÀY ---
    # 1. Kiểm tra và load checkpoint (nếu có)
    state_path = os.path.join(config["out_dir"], "training_state.pt")
    if os.path.exists(state_path):
        # load_checkpoint sẽ load weights cho unet, text_encoder VÀ unfreeze nếu cần
        start_epoch, best_val_loss, optimizer_state = load_checkpoint(
            unet, text_encoder, config["out_dir"], device
        )
    
    # 2. TẠO optimizer SAU KHI đã load model
    # (vì text_encoder có thể đã được unfreeze, make_optimizer sẽ tạo đúng số group)
    optimizer = make_optimizer(unet, text_encoder, config["lr"])
    
    # 3. LOAD state cho optimizer
    if optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Optimizer state loaded successfully.")
        except ValueError as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Starting optimizer from scratch.")
    # --- KẾT THÚC PHẦN THAY THẾ ---
    
    global_step = 0
    for epoch in range(start_epoch, config["epochs"]):

        # Optional Phase 2: unfreeze CLIP mid-training
        # --- BỎ KHỐI IF NÀY ---
        # (Vì logic unfreeze đã được chuyển vào load_checkpoint)
        # if epoch == PHASE1_EPOCHS and N_TEXT_UNFREEZE > 0:
        #     print(">>> Phase 2: unfreezing last", N_TEXT_UNFREEZE, "layers of text encoder.")
        #     set_text_encoder_trainable(text_encoder, n_unfreeze=N_TEXT_UNFREEZE)
        #     optimizer = make_optimizer(unet, text_encoder, config["lr"])

        unet.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # encode text (only grad if unfreezed)
            if any(p.requires_grad for p in text_encoder.parameters()):
                encoder_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    encoder_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden_states = encoder_outputs.last_hidden_state

            latents = encode_images_to_latents(vae, pixel_values).detach()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps,
                                      (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{running_loss/global_step:.6f}"})

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg train loss: {avg_train_loss:.6f}")

        # ------------------ VALIDATION ------------------
        val_loss = validate_one_epoch(vae, unet, text_encoder, tokenizer, noise_scheduler, val_loader, device)
        print(f"Epoch {epoch+1} val_loss: {val_loss:.6f}")
        
        # ------------------ SAMPLE EVALUATION (fixed) ------------------
        # We'll sample K captions from val, load their GT images, generate images for those captions,
        # then compute CLIP, LPIPS, FID between generated images and GT sketches.
        K = 8
        val_text_dir = os.path.join(DATA_ROOT, "val", "texts")
        val_img_dir = os.path.join(DATA_ROOT, "val", "images")
        val_text_paths = sorted(glob(os.path.join(val_text_dir, "*.txt")))
        if len(val_text_paths) == 0:
            # fallback: use generate_samples_for_eval (older behavior) but metrics not computed
            sample_imgs, sample_txts = generate_samples_for_eval(unet, vae, text_encoder, tokenizer, noise_scheduler, device, K, SEED+epoch)
            clip_score = metric.clip_text_image_score(sample_imgs, sample_txts, device)
            lpips_score = None
            fid_score = None
            print(f"Epoch {epoch+1} sample CLIP-score: {clip_score:.4f} (no GT available for LPIPS/FID)")
        else:
            selected = random.sample(val_text_paths, min(K, len(val_text_paths)))
            captions = []
            gt_tensors = []
            from torchvision.transforms import ToTensor
            to_tensor = ToTensor()
            for p in selected:
                # read caption
                with open(p, "r", encoding="utf-8") as f:
                    captions.append(f.readline().strip())
                # construct image path (matching naming convention)
                name = os.path.splitext(os.path.basename(p))[0]
                # try common extensions
                img_path_png = os.path.join(val_img_dir, name + ".png")
                img_path_jpg = os.path.join(val_img_dir, name + ".jpg")
                if os.path.exists(img_path_png):
                    img_path = img_path_png
                elif os.path.exists(img_path_jpg):
                    img_path = img_path_jpg
                else:
                    # fallback: try any matching file
                    candidates = glob(os.path.join(val_img_dir, name + ".*"))
                    if len(candidates) > 0:
                        img_path = candidates[0]
                    else:
                        # if not found, skip this sample
                        continue
                img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                gt_tensors.append(to_tensor(img))  # in [0,1]

            # generate images for captions (use same sampling as in generate_samples_for_eval but deterministic per caption)
            gen_images = []
            # get latent shape safely
            C, H_lat, W_lat, scaling = get_latent_shape_from_vae(vae, IMAGE_SIZE, device)
            noise_scheduler.set_timesteps(num_inference_steps := 50, device=device)
            unet.eval()
            text_encoder.eval()
            for cap in captions[:len(gt_tensors)]:  # ensure same length
                tokenized = tokenizer(cap, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
                encoder_hidden_states = text_encoder(**tokenized).last_hidden_state

                # init latents matching unet dtype/device
                latents = torch.randn((1, C, H_lat, W_lat), device=next(unet.parameters()).device, dtype=next(unet.parameters()).dtype)
                if scaling is not None and scaling != 1.0:
                    latents = latents * scaling

                for t in noise_scheduler.timesteps:
                    noise_pred = unet(latents, t, encoder_hidden_states).sample
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                # decode and convert to [0,1] tensor
                if hasattr(vae, "decode"):
                    imgs = vae.decode(latents / (scaling if scaling is not None else 1.0)).sample
                else:
                    imgs = vae.decode(latents).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                gen_images.append(imgs[0].cpu())

            # ensure lengths match
            if len(gen_images) == 0 or len(gt_tensors) == 0:
                clip_score = 0.0
                lpips_score = None
                fid_score = None
                print(f"Epoch {epoch+1}: could not generate/locate samples for evaluation.")
            else:
                # stacks and move to device for metric computation
                gen_stack = torch.stack(gen_images).to(device)   # [B,3,H,W], in [0,1]
                gt_stack = torch.stack(gt_tensors).to(device)   # [B,3,H,W], in [0,1]

                clip_score = metric.clip_text_image_score(list(gen_stack.cpu()), captions[:gen_stack.shape[0]], device)
                lpips_score = metric.lpips(gen_stack, gt_stack)
                fid_score = metric.fid(gen_stack, gt_stack)
                print(f"Epoch {epoch+1} sample CLIP-score: {clip_score:.4f}, LPIPS: {lpips_score:.4f}, FID: {fid_score:.4f}")

        # ------------------ SAVE MODEL ------------------
        # save_checkpoint(unet, optimizer, epoch, best_val_loss, config["out_dir"])
        save_checkpoint(unet, text_encoder, optimizer, epoch, best_val_loss, config["out_dir"])
         
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config["out_dir"], "best_unet")
            unet.save_pretrained(save_path)
            print("Saved best_unet at", save_path)
            
            # --- Save text encoder if unfrozen ---
            if epoch >= PHASE1_EPOCHS and N_TEXT_UNFREEZE > 0:
                save_path_text_encoder = os.path.join(config["out_dir"], "best_text_encoder")
                text_encoder.save_pretrained(save_path_text_encoder)
                print("Saved best_text_encoder at", save_path_text_encoder)
            
        if (epoch + 1) % config["save_every"] == 0:
            save_path_unet = os.path.join(config["out_dir"], f"unet_epoch{epoch+1}")
            unet.save_pretrained(save_path_unet)
            
            if epoch >= PHASE1_EPOCHS and N_TEXT_UNFREEZE > 0:
                save_path_text_encoder = os.path.join(config["out_dir"], f"text_encoder_epoch{epoch+1}")
                text_encoder.save_pretrained(save_path_text_encoder)

            print(f"Checkpoint saved at epoch {epoch+1}.")
        
        # ------------------ LOG METRICS TO CSV ------------------
        import pandas as pd
        metrics_path = os.path.join(config["out_dir"], "metrics_log.csv")
        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "CLIP_score": clip_score if 'clip_score' in locals() else None,
            "LPIPS": lpips_score if 'lpips_score' in locals() else None,
            "FID": fid_score if 'fid_score' in locals() else None
        }
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(metrics_path, index=False)
        print(f"Epoch {epoch+1} metrics saved to {metrics_path}")
        # ------------------ SAVE SEPARATE METRICS ------------------
        unet_metrics_path = os.path.join(config["out_dir"], "unet_metrics.csv")
        clip_metrics_path = os.path.join(config["out_dir"], "clip_metrics.csv")

        unet_row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "LPIPS": lpips_score if 'lpips_score' in locals() else None,
            "FID": fid_score if 'fid_score' in locals() else None
        }
        clip_row = {
            "epoch": epoch + 1,
            "CLIP_score": clip_score if 'clip_score' in locals() else None
        }

        # --- Ghi file UNet metrics ---
        if os.path.exists(unet_metrics_path):
            df_u = pd.read_csv(unet_metrics_path)
            df_u = pd.concat([df_u, pd.DataFrame([unet_row])], ignore_index=True)
        else:
            df_u = pd.DataFrame([unet_row])
        df_u.to_csv(unet_metrics_path, index=False)

        # --- Ghi file CLIP metrics ---
        if os.path.exists(clip_metrics_path):
            df_c = pd.read_csv(clip_metrics_path)
            df_c = pd.concat([df_c, pd.DataFrame([clip_row])], ignore_index=True)
        else:
            df_c = pd.DataFrame([clip_row])
        df_c.to_csv(clip_metrics_path, index=False)
        print(f"Epoch {epoch+1} metrics saved separately: UNet({unet_metrics_path}), CLIP({clip_metrics_path})")
        
    print("Training finished.")


def save_checkpoint(unet, text_encoder, optimizer, epoch, best_val_loss, out_dir):
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "optimizer": optimizer.state_dict(),
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, "training_state.pt"))
    unet.save_pretrained(os.path.join(out_dir, "unet_last"))
    
    # --- THÊM PHẦN NÀY ---
    # Kiểm tra xem text_encoder có tham số đang train không
    if any(p.requires_grad for p in text_encoder.parameters()):
        text_encoder.save_pretrained(os.path.join(out_dir, "text_encoder_last"))
        print("Saved text_encoder_last.")
    # --- KẾT THÚC ---
        
    print(f"Saved checkpoint at epoch {epoch+1} to {out_dir}")

def load_checkpoint(unet, text_encoder, out_dir, device): # Thêm text_encoder, bỏ optimizer
    state_path = os.path.join(out_dir, "training_state.pt")
    if os.path.exists(state_path):
        print(f"Resuming from checkpoint: {state_path}")
        state = torch.load(state_path, map_location=device)
        
        # Load UNet
        weights_path = os.path.join(out_dir, "unet_last", "diffusion_pytorch_model.safetensors")
        unet.load_state_dict(load_file(weights_path, device=device))
        
        # Load Text Encoder NẾU CÓ
        text_encoder_weights_path = os.path.join(out_dir, "text_encoder_last", "model.safetensors")
        if not os.path.exists(text_encoder_weights_path):
             # Thử tìm file .bin (thay thế nếu bạn dùng format cũ)
             text_encoder_weights_path = os.path.join(out_dir, "text_encoder_last", "pytorch_model.bin")

        if os.path.exists(text_encoder_weights_path):
            print("Resuming text_encoder_last weights.")
            if text_encoder_weights_path.endswith(".safetensors"):
                text_encoder.load_state_dict(load_file(text_encoder_weights_path, device=device))
            else:
                text_encoder.load_state_dict(torch.load(text_encoder_weights_path, map_location=device))
                
            # QUAN TRỌNG: Unfreeze nó ngay
            set_text_encoder_trainable(text_encoder, n_unfreeze=N_TEXT_UNFREEZE)
            print(f"Unfroze text encoder (resuming from Phase 2).")

        # Trả về state của optimizer để xử lý sau
        return state["epoch"] + 1, state["best_val_loss"], state["optimizer"]
    else:
        print("No checkpoint found — starting from scratch.")
        return 0, float("inf"), None # Trả về None
    
@torch.no_grad()
def validate_one_epoch(vae, unet, text_encoder, tokenizer, noise_scheduler, val_loader, device):
    unet.eval()
    total, n = 0.0, 0
    for batch in tqdm(val_loader, desc="Validation"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        enc_out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        hidden = enc_out.last_hidden_state
        latents = encode_images_to_latents(vae, pixel_values).detach()
        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, t)
        pred = unet(noisy, t, hidden).sample
        loss = F.mse_loss(pred, noise)
        total += loss.item() * latents.size(0)
        n += latents.size(0)
    return total / max(1, n)

# ---------------------------
# Generation helper
# ---------------------------
@torch.no_grad()
def get_latent_shape_from_vae(vae, image_size, device):
    """
    Return (C, H_lat, W_lat) latent channels and spatial dims by encoding a dummy image.
    """
    vae_device = next(vae.parameters()).device
    dummy = torch.zeros((1, 3, image_size, image_size), device=vae_device, dtype=torch.float32)
    # vae expects inputs in [-1,1]
    dummy = dummy * 2.0 - 1.0
    enc = vae.encode(dummy)
    # latent sample shape: [B, C, H, W] (latent_dist.sample() or .mean())
    lat = enc.latent_dist.sample()
    C = lat.shape[1]
    H = lat.shape[2]
    W = lat.shape[3]
    return C, H, W, getattr(vae.config, "scaling_factor", 1.0)

@torch.no_grad()
def generate_samples_for_eval(unet, vae, text_encoder, tokenizer, noise_scheduler, device, sample_count=8, seed=0, num_inference_steps=50):
    torch.manual_seed(seed)
    unet.eval()
    text_encoder.eval()

    # load sample captions
    val_text_paths = sorted(glob(os.path.join(DATA_ROOT, "val", "texts", "*.txt")))
    selected = random.sample(val_text_paths, min(sample_count, len(val_text_paths)))
    captions = []
    for p in selected:
        with open(p, "r", encoding="utf-8") as f:
            captions.append(f.readline().strip())

    # get accurate latent shape from VAE
    C, H_lat, W_lat, scaling = get_latent_shape_from_vae(vae, IMAGE_SIZE, device)
    generated_images = []

    # prepare scheduler timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    for cap in captions:
        tokenized = tokenizer(cap, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        encoder_hidden_states = text_encoder(**tokenized).last_hidden_state

        # create initial latents on same device/dtype as UNet/vae
        latents = torch.randn((1, C, H_lat, W_lat), device=next(unet.parameters()).device, dtype=next(unet.parameters()).dtype)
        # scale latents if vae uses scaling_factor
        if scaling is not None and scaling != 1.0:
            latents = latents * scaling

        # denoising loop (DDPM/PNDM via scheduler)
        for t in tqdm(noise_scheduler.timesteps, leave=False):
            # ensure timestep tensor dtype matches scheduler expectation
            step_output = unet(latents, t, encoder_hidden_states)
            noise_pred = step_output.sample
            step = noise_scheduler.step(noise_pred, t, latents)
            latents = step.prev_sample

        # decode latents to images
        if hasattr(vae, "decode"):
            imgs = vae.decode(latents / (scaling if scaling is not None else 1.0)).sample
        else:
            imgs = vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        img_tensor = imgs[0].cpu()
        generated_images.append(img_tensor)

    return generated_images, captions


# small helper used in training code placeholder (not used)
def pil_to_tensor_from_textref(pil_img, device):
    # convert PIL to tensor [C,H,W] in [0,1]
    if isinstance(pil_img, torch.Tensor):
        return pil_img
    tf = transforms.ToTensor()
    return tf(pil_img).to(device)


# ---------------------------
# Main
# ---------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = DEVICE
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_components()

    # datasets + loaders
    train_ds = TextSketchDataset(os.path.join(DATA_ROOT, "train"), tokenizer, image_size=IMAGE_SIZE)
    val_ds = TextSketchDataset(os.path.join(DATA_ROOT, "val"), tokenizer, image_size=IMAGE_SIZE)
    test_ds = TextSketchDataset(os.path.join(DATA_ROOT, "test"), tokenizer, image_size=IMAGE_SIZE)

    # optional quick debug subset
    if TRAIN_SUBSET_FRAC < 1.0:
        import math
        keep_n = max(1, int(len(train_ds) * TRAIN_SUBSET_FRAC))
        train_ds.image_files = train_ds.image_files[:keep_n]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    config = {
        "lr": LR,
        "epochs": NUM_EPOCHS,
        "out_dir": OUTPUT_DIR,
        "save_every": SAVE_EVERY
    }

    train_loop(vae, unet, text_encoder, tokenizer, noise_scheduler, train_loader, val_loader, device, config)

    # After training, run final evaluation on test set
    print("Running final evaluation on test set (using test_loader)...")
    metric = Metrics(device)
    
    # Đảm bảo các model ở chế độ eval()
    unet.eval()
    vae.eval()
    text_encoder.eval()
    
    all_lpips_scores = []
    all_clip_scores = []
    
    # 1. Reset FID metric trước khi bắt đầu
    metric.fid_metric.reset()
    
    # Lấy thông số latent một lần
    C, H_lat, W_lat, scaling = get_latent_shape_from_vae(vae, IMAGE_SIZE, device)
    noise_scheduler.set_timesteps(num_inference_steps=50, device=device)

    # 2. Sử dụng test_loader (đã được tạo ở trên)
    for batch in tqdm(test_loader, desc="Batched Test Evaluation"):
        # Lấy dữ liệu thật từ batch
        real_images = batch["pixel_values"].to(device) # Ảnh thật [B, 3, H, W]
        captions = batch["captions"]                   # List các caption [B]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        current_batch_size = real_images.shape[0]

        # --- Tạo ảnh giả (theo BATCH) ---
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            
            latents = torch.randn((current_batch_size, C, H_lat, W_lat), 
                                  device=device, dtype=unet.dtype)
            if scaling is not None and scaling != 1.0:
                latents = latents * scaling

            for t in noise_scheduler.timesteps:
                noise_pred = unet(latents, t, encoder_hidden_states).sample
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # generated_batch ở dạng [0, 1]
            generated_batch = vae.decode(latents / (scaling if scaling is not None else 1.0)).sample
            generated_batch = (generated_batch / 2 + 0.5).clamp(0, 1)
        # --- Kết thúc tạo ảnh (nhanh hơn nhiều) ---

        # 3. Tính toán và tích lũy metrics
        
        # LPIPS (tính trung bình của batch)
        batch_lpips = metric.lpips(generated_batch, real_images) 
        all_lpips_scores.append(batch_lpips) # Lưu ý: hàm lpips của bạn đã trả về .mean()
        
        # CLIP (tính trung bình của batch)
        # Chuyển tensor về list ảnh PIL/tensor CPU cho hàm CLIP
        gen_list_cpu = [img for img in generated_batch.cpu()] 
        batch_clip = metric.clip_text_image_score(gen_list_cpu, captions, device)
        all_clip_scores.append(batch_clip)

        # FID (cập nhật tích lũy, KHÔNG tính ngay)
        real_images_uint8 = (real_images * 255).byte()
        generated_batch_uint8 = (generated_batch * 255).byte()
        
        metric.fid_metric.update(real_images_uint8, real=True)
        metric.fid_metric.update(generated_batch_uint8, real=False)

    # 4. Tính kết quả cuối cùng sau khi lặp xong
    
    # LPIPS và CLIP: Lấy trung bình của các điểm số trung bình của từng batch
    final_lpips = torch.tensor(all_lpips_scores).mean().item()
    final_clip_score = torch.tensor(all_clip_scores).mean().item()
    
    # FID: Tính toán 1 LẦN DUY NHẤT sau khi đã "update" tất cả các batch
    final_fid = metric.fid_metric.compute().item()

    print(f"Final Test LPIPS={final_lpips:.4f}, CLIP-score={final_clip_score:.4f}, FID={final_fid:.4f}")

    # 5. Lưu kết quả
    metrics_path = os.path.join(OUTPUT_DIR, "final_test_metrics.csv") # Đổi tên file để tránh ghi đè
    row = {
        "Epoch": "Final", # Thêm cột để phân biệt
        "LPIPS": final_lpips,
        "CLIP_score": final_clip_score,
        "FID": final_fid
    }
    df = pd.DataFrame([row])
    df.to_csv(metrics_path, index=False)
    print(f"Final metrics saved to {metrics_path}")
    
    # Lấy batch cuối cùng để lưu mẫu
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)
    for i, img in enumerate(generated_batch[:16]): # Lưu 16 ảnh từ batch cuối cùng
        utils.save_image(img, os.path.join(OUTPUT_DIR, "samples", f"sample_{i:03d}.png"))

    print("Done. Samples saved to", os.path.join(OUTPUT_DIR, "samples"))

def plot_training_curves(output_dir):
    unet_metrics_path = os.path.join(output_dir, "unet_metrics.csv")
    clip_metrics_path = os.path.join(output_dir, "clip_metrics.csv")

    if os.path.exists(unet_metrics_path):
        df_u = pd.read_csv(unet_metrics_path)
        plt.figure(figsize=(10,5))
        plt.plot(df_u["epoch"], df_u["train_loss"], label="Train Loss")
        plt.plot(df_u["epoch"], df_u["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("UNet Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "unet_loss_curve.png"))
        plt.close()

        plt.figure(figsize=(10,5))
        plt.plot(df_u["epoch"], df_u["LPIPS"], label="LPIPS")
        plt.plot(df_u["epoch"], df_u["FID"], label="FID")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("UNet LPIPS & FID")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "unet_quality_curve.png"))
        plt.close()

    if os.path.exists(clip_metrics_path):
        df_c = pd.read_csv(clip_metrics_path)
        plt.figure(figsize=(10,5))
        plt.plot(df_c["epoch"], df_c["CLIP_score"], label="CLIP-score", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("CLIP-score")
        plt.title("CLIP Semantic Alignment Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "clip_score_curve.png"))
        plt.close()

    print("Training curves saved to:", output_dir)
    
if __name__ == "__main__":
    main()
    plot_training_curves(OUTPUT_DIR)