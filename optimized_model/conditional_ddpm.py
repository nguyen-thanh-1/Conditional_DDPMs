# conditional_ddpm.py (Phiên bản LDM tối ưu cho CPU - Đã sửa lỗi num_res_blocks)
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
import time
from tqdm import tqdm 

try:
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
    from diffusers.models import UNet2DConditionModel, AutoencoderKL
except ImportError:
    print("Lỗi: Vui lòng cài đặt diffusers. Chạy: pip install diffusers")
    exit()

from Conditioning import CLIPTextEmbedder, ConditionalGaussianDiffusion
from Trainer import TextImageDataset, LatentCaptionDataset, Trainer
from Evaluation_Logging import Evaluator, TrainingLogger, Metrics

# (Các phần khác giữ nguyên)
# ...

def training(
    root_dir: str = "dataset",
    cache_dir: str = "latents_cache",
    ckpt_resume_path: str = 'ckpt_last.pt',
    timesteps: int = 1000, 
    image_size: int = 32,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    sampling_steps: int = 25,
    guidance_scale: float = 7.5,
    vae_repo_name: str = "runwayml/stable-diffusion-v1-5",
    clip_repo_name: str = "openai/clip-vit-large-patch14"
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for LDM Training")

    # (Phần Dataset và VAE giữ nguyên)
    # ... (Giống hệt file trước)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: 2.0 * t - 1.0) # -> [-1,1]
    ])
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset folder '{root_dir}' not found.")
    
    img_folder = datasets.ImageFolder(root=root_dir, transform=None)
    if len(img_folder) == 0:
        raise RuntimeError(f"No images found in '{root_dir}'.")
    
    image_paths = [path for path, _ in img_folder.samples]
    captions = [img_folder.classes[label] for _, label in img_folder.samples]

    print(f"Loading pre-trained VAE ({vae_repo_name})...")
    vae = AutoencoderKL.from_pretrained(
        vae_repo_name, 
        subfolder="vae"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    LATENT_Z_CHANNELS = vae.config.latent_channels
    vae_scaling_factor = vae.config.scaling_factor 
    LATENT_IMAGE_SIZE = image_size // (2 ** (len(vae.config.block_out_channels) - 1))
    print(f"Pre-trained VAE loaded. Latent channels: {LATENT_Z_CHANNELS}, Latent size: {LATENT_IMAGE_SIZE}, Scale factor: {vae_scaling_factor:.4f}")

    os.makedirs(cache_dir, exist_ok=True)
    expected_last_file = os.path.join(cache_dir, f"latent_{len(captions) - 1}.pt")
    
    if not os.path.exists(expected_last_file):
        print(f"Cache not found or incomplete. Starting latent precaching to '{cache_dir}'...")
        caching_dataset = TextImageDataset(image_paths, captions, transform=transform)
        caching_loader = DataLoader(caching_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=0, pin_memory=False)
        
        idx_counter = 0
        with torch.no_grad():
            for imgs_batch, _ in tqdm(caching_loader, desc="Precaching latents"):
                imgs_batch = imgs_batch.to(device)
                latent_dist = vae.encode(imgs_batch).latent_dist
                z = latent_dist.sample()
                z = z * vae_scaling_factor
                
                z_cpu = z.cpu()
                for i in range(z_cpu.shape[0]):
                    save_path = os.path.join(cache_dir, f"latent_{idx_counter}.pt")
                    torch.save(z_cpu[i], save_path)
                    idx_counter += 1
        
        print(f"Precaching complete. {idx_counter} latents saved.")
    else:
        print(f"Found existing cache in '{cache_dir}'. Skipping precaching.")

    dataset = LatentCaptionDataset(cache_dir, captions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    print(f"Loaded {len(dataset)} cached latents for training.")
    
    # -------------------------
    # Model (ĐÃ SỬA)
    # -------------------------
    print("Initializing UNet2DConditionModel from diffusers...")
    model = UNet2DConditionModel(
        sample_size=LATENT_IMAGE_SIZE,
        in_channels=LATENT_Z_CHANNELS,
        out_channels=LATENT_Z_CHANNELS,
        block_out_channels=(128, 256, 512), 
        down_block_types=(
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D", 
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D"
        ),
        cross_attention_dim=768, 
        # num_res_blocks=2, # <-- SỬA: Đã xóa dòng này
        attention_head_dim=8,
    ).to(device)
    
    text_embedder = CLIPTextEmbedder(pretrained=clip_repo_name, device=device)

    # (Phần Diffusion, Scheduler, Metrics, Trainer... giữ nguyên)
    # ... (Giống hệt file trước)
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        text_embedder=text_embedder,
        image_size=LATENT_IMAGE_SIZE,
        channels=LATENT_Z_CHANNELS,
        timesteps=timesteps,
        beta_schedule='linear',
        device=device
    )

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="linear"
    )

    metrics = Metrics(device=device)
    n_val = min(16, len(image_paths))
    real_dataset_for_eval = TextImageDataset(image_paths, captions, transform=transform)
    real_images_list = [real_dataset_for_eval[i][0] for i in range(n_val)]
    real_images = torch.stack(real_images_list, dim=0)
    real_images = (real_images + 1.0) / 2.0 # [0,1]
    prompts_for_eval = [captions[i] for i in range(n_val)]
    evaluator = Evaluator(metrics, prompts_for_eval, real_images)
    logger = TrainingLogger(logdir="logs", use_wandb=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        vae=vae, 
        vae_scaling_factor=vae_scaling_factor,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader, 
        evaluator=evaluator,
        logger=logger,
        device=device,
        sampling_steps=sampling_steps,
        guidance_scale=guidance_scale
    )

    ckpt_dir = "checkpoints_quickdraw_ldm"
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = os.path.join(ckpt_dir, ckpt_resume_path)
    start_epoch = 0
    
    if os.path.exists(resume_path):
        try:
            start_epoch = trainer.load_checkpoint(resume_path)
        except Exception as e:
            print(f"Warning: Could not load checkpoint '{resume_path}'. Starting from scratch. Error: {e}")
            start_epoch = 0
    else:
        print("Starting training from scratch")
        
    end_epoch = start_epoch + epochs

    for epoch in range(start_epoch, end_epoch):
        print(f"=== Epoch {epoch}/{end_epoch-1} ===")
        avg_loss = trainer.train_one_epoch(epoch)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == end_epoch:
            ckpt_path_epoch = os.path.join(ckpt_dir, f"ckpt_epoch{epoch+1}.pt")
            trainer.save_checkpoint(ckpt_path_epoch, epoch=epoch+1)

    ckpt_path_last = os.path.join(ckpt_dir, "ckpt_last.pt")
    trainer.save_checkpoint(ckpt_path_last, epoch=end_epoch)
    print(f'Finished training. Final checkpoint saved to {ckpt_path_last}')


def inference(
    ckpt_path: str = "checkpoints_quickdraw_ldm/ckpt_last.pt",
    prompt: str = "cat",
    out_file: str = "sample.png",
    device: Optional[torch.device] = None,
    timesteps: int = 1000,
    sampling_steps: int = 25,
    guidance_scale: float = 7.5,
    vae_repo_name: str = "runwayml/stable-diffusion-v1-5",
    clip_repo_name: str = "openai/clip-vit-large-patch14",
    image_size: int = 32
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for LDM Inference")
    use_amp = (device.type == 'cuda') 

    # (Phần VAE load giữ nguyên)
    # ...
    print(f"Loading pre-trained VAE ({vae_repo_name})...")
    vae = AutoencoderKL.from_pretrained(
        vae_repo_name, 
        subfolder="vae"
    ).to(device)
    vae.eval()
    
    LATENT_Z_CHANNELS = vae.config.latent_channels
    vae_scaling_factor = vae.config.scaling_factor
    LATENT_IMAGE_SIZE = image_size // (2 ** (len(vae.config.block_out_channels) - 1))
    print("Pre-trained VAE loaded for decoding.")

    # -------------------------
    # 2. Khởi tạo UNet (ĐÃ SỬA)
    # -------------------------
    model = UNet2DConditionModel(
        sample_size=LATENT_IMAGE_SIZE,
        in_channels=LATENT_Z_CHANNELS,
        out_channels=LATENT_Z_CHANNELS,
        block_out_channels=(128, 256, 512), 
        down_block_types=(
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D", 
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D"
        ),
        cross_attention_dim=768,
        # num_res_blocks=2, # <-- SỬA: Đã xóa dòng này
        attention_head_dim=8,
    ).to(device)
    
    text_embedder = CLIPTextEmbedder(pretrained=clip_repo_name, device=device)

    # (Phần còn lại của inference giữ nguyên)
    # ... (Giống hệt file trước)
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        text_embedder=text_embedder,
        image_size=LATENT_IMAGE_SIZE,
        channels=LATENT_Z_CHANNELS,
        timesteps=timesteps,
        device=device
    )

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="linear"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Dummy
    
    trainer = Trainer(
        model=model, diffusion=diffusion, vae=vae, optimizer=optimizer,
        scheduler=scheduler, dataloader=None, evaluator=None, logger=None, 
        device=device, vae_scaling_factor=vae_scaling_factor
    )
    if os.path.exists(ckpt_path):
        trainer.load_checkpoint(ckpt_path)
        print(f"Loaded LDM checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No LDM checkpoint found at {ckpt_path}")

    print(f"Generating image for prompt: '{prompt}'...")
    
    ema_model = trainer.ema.module.to(device)
    ema_model.eval()
    
    B = 1 
    start_time = time.time()
    with torch.no_grad():
        _, context = text_embedder.encode([prompt] * B)
        context = context.to(device)
        null_context = diffusion.null_context.repeat(B, 1, 1)

        scheduler.set_timesteps(sampling_steps)
        z_shape = (B, diffusion.channels, diffusion.image_size, diffusion.image_size)
        latents = torch.randn(z_shape, device=device)
        
        device_type = 'cuda' if use_amp else 'cpu'
        with torch.autocast(device_type=device_type, enabled=use_amp):
            for t in scheduler.timesteps: 
                latent_model_input = torch.cat([latents] * 2)
                t_in = t.repeat(B * 2).to(device)
                context_in = torch.cat([null_context, context])

                noise_pred_all = ema_model(
                    sample=latent_model_input, 
                    timestep=t_in, 
                    encoder_hidden_states=context_in
                ).sample
                
                noise_pred_uncond, noise_pred_cond = noise_pred_all.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        latents_unscaled = latents / vae_scaling_factor
        samples = vae.decode(latents_unscaled).sample

    end_time = time.time()
    print(f"Inference took {end_time - start_time:.2f} seconds.")

    out = (samples + 1.0) / 2.0
    out = torch.clamp(out, 0.0, 1.0)
    save_image(out, out_file)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    
    mode = 'train_ldm'
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    my_dataset_path = r'C:\Users\PC\OneDrive\Desktop\scientific research\conditional DDPM\dataset_test\test_bug'
    my_cache_path = r'C:\Users\PC\OneDrive\Desktop\scientific research\conditional DDPM\latents_cache_test_bug' 
    
    if not os.path.isdir(my_dataset_path):
        print(f"Warning: Dataset path '{my_dataset_path}' not found. Please update 'my_dataset_path'.")
        exit()

    if mode == "train_ldm":
        print("--- Mode: Training LDM (CPU Optimized) ---")
        training(
            root_dir=my_dataset_path,
            cache_dir=my_cache_path, 
            ckpt_resume_path='ckpt_last.pt',
            timesteps=1000,
            epochs=100,
            batch_size=16,
            device=dev,
            sampling_steps=25,
            guidance_scale=7.5
        )
            
    elif mode == "inference":
        print("--- Mode: Inference (CPU) ---")
        inference(
            ckpt_path=r"checkpoints_quickdraw_ldm/ckpt_last.pt", 
            prompt="apple", 
            out_file="apple_ldm_cpu.png",
            device=dev,
            timesteps=1000,
            sampling_steps=25,
            guidance_scale=7.5
        )
    else:
        print(f"Unknown mode: {mode}. Available modes: 'train_ldm', 'inference'")