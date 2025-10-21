# conditional_ddpm_fixed.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from typing import Optional
import torch.nn.functional as F

from dataclasses import dataclass

# Import các module đã được tối ưu hóa
from Conditioning import ConditionalUNet, CLIPTextEmbedder, ConditionalGaussianDiffusion
from Latent_Diffusion import SimpleVAE  # Chỉ import VAE
from Guidance_Sampling import DDIMSampler, SamplerConfig
from Trainer import TextImageDataset, Trainer, benchmark_text_encoder, benchmark_iterations, benchmark_images  # Import Dataset và Trainer đã sửa
from Evaluation_Logging import Evaluator, TrainingLogger, Metrics

# --- ĐỊNH NGHĨA THÔNG SỐ LDM (MỚI) ---
# Các thông số này PHẢI KHỚP với SimpleVAE đã huấn luyện của bạn
LATENT_Z_CHANNELS = 4  # Số kênh của latent z (từ VAE)
LATENT_IMAGE_SIZE = 4  # Kích thước không gian của latent z (ví dụ: 32x32 -> 4x4)
VAE_CHECKPOINT_PATH = "vae_pretrained.pt" # Đường dẫn đến VAE đã huấn luyện
# ---
# ----------------------------------------
# === CÁC HÀM ĐỂ HUẤN LUYỆN VAE (MỚI) ===
# ----------------------------------------

@dataclass
class VAEConfig:
    lr: float = 1e-4
    epochs: int = 50 # Tăng epochs để VAE tốt hơn
    batch_size: int = 64
    recon_weight: float = 1.0
    kl_weight: float = 0.0001

    base_ch: int = 128           # <-- Tăng khả năng VAE (từ 64)
    kl_warmup_epochs: int = 10

    dataset_root: str = './data'
    save_path: str = './vae_ckpts'

def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, recon_weight: float = 1.0, kl_weight: float = 0.01) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x)
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Chuẩn hóa KL loss (chia cho số pixel)
    kl = kl / (recon.shape[0] * recon.shape[1] * recon.shape[2] * recon.shape[3])
    return recon_weight * recon_loss + kl_weight * kl

def run_vae_training(
    image_size: int = 32, 
    config: VAEConfig = VAEConfig(),
    device: Optional[torch.device] = None
):
    """
    Hàm độc lập để huấn luyện SimpleVAE.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting VAE training on device: {device}")
    
    vae = SimpleVAE(in_channels=3, base_ch=64, z_channels=LATENT_Z_CHANNELS).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=config.lr)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # [-1, 1]
    ])
    
    # Sử dụng dataset từ thư mục của bạn thay vì CIFAR10 nếu muốn
    # --- THAY ĐỔI CỐT LÕI ---
    # Sử dụng ImageFolder để tải dataset của bạn
    if not os.path.isdir(config.dataset_root):
        raise FileNotFoundError(f"VAE training error: Dataset folder '{config.dataset_root}' not found.")
        
    ds = datasets.ImageFolder(root=config.dataset_root, transform=transform)
    
    if len(ds) == 0:
        raise RuntimeError(f"No images found in '{config.dataset_root}'. Please check the path and folder structure.")
        
    print(f"Loaded ImageFolder dataset for VAE training with {len(ds)} samples from '{config.dataset_root}'.")
    # --- KẾT THÚC THAY ĐỔI ---
    
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    os.makedirs(config.save_path, exist_ok=True)

    best_loss = float('inf')
    for epoch in range(config.epochs):
        vae.train()
        running_loss = 0.0
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar, config.recon_weight, config.kl_weight)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{config.epochs} | Step {i+1}/{len(loader)} | Loss: {running_loss/100:.5f}")
                running_loss = 0.0
        
        avg_loss = running_loss / len(loader) if len(loader) > 0 else 0

        # Lưu checkpoint và sample reconstructions
        ckpt_path = os.path.join(config.save_path, f'vae_epoch_{epoch+1}.pt')
        torch.save({'vae_state_dict': vae.state_dict(), 'epoch': epoch+1}, ckpt_path)

        # Lưu checkpoint tốt nhất (sẽ được đổi tên thành VAE_CHECKPOINT_PATH)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt_path = os.path.join(config.save_path, "vae_best.pt")
            torch.save({'vae_state_dict': vae.state_dict(), 'epoch': epoch+1}, best_ckpt_path)
            print(f"New best VAE model saved to {best_ckpt_path} (Loss: {avg_loss:.5f})")

        # sample reconstructions
        vae.eval()
        with torch.no_grad():
            xs = next(iter(loader))[0][:16].to(device)
            recon, _, _ = vae(xs)
            grid = torch.cat([xs, recon], dim=0)
            grid = (grid + 1.0) / 2.0 # Về [0, 1]
            save_image(grid, os.path.join(config.save_path, f'recon_epoch_{epoch+1}.png'), nrow=8)

    print('VAE training finished.')
    print(f"Best VAE checkpoint saved at: {os.path.join(config.save_path, 'vae_best.pt')}")
    print(f"Please rename 'vae_best.pt' to '{VAE_CHECKPOINT_PATH}' to use it for LDM training.")

# ----------------------------------------
# === KẾT THÚC PHẦN CODE VAE ===
# ----------------------------------------
def training(
    root_dir="dataset",
    ckpt_resume_path = 'ckpt_last.pt',
    timesteps = 500, # Giảm timesteps cho LDM
    image_size=32, # Kích thước ảnh pixel gốc
    batch_size=8,
    add_epochs = 0,
    epochs=100,
    lr=1e-4,
    device=None,
):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device} for LDM Training")

    # -------------------------
    # Dataset (TỐI ƯU HÓA I/O)
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),               # [0,1]
        transforms.Lambda(lambda t: 2.0 * t - 1.0)  # -> [-1,1]
    ])

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset folder '{root_dir}' not found.")

    # 1. Không transform ở đây
    img_folder = datasets.ImageFolder(root=root_dir, transform=None)
    if len(img_folder) == 0:
        raise RuntimeError(f"No images found in '{root_dir}'.")

    # 2. Chỉ lấy paths và captions (Tối ưu RAM)
    image_paths = [path for path, _ in img_folder.samples]
    captions = [img_folder.classes[label] for _, label in img_folder.samples]

    # 3. Dataset sẽ tải và transform on-the-fly
    dataset = TextImageDataset(image_paths, captions, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Loaded {len(dataset)} image paths for on-the-fly training.")

    # -------------------------
    # Model / Diffusion / VAE (TỐI ƯU HÓA LDM)
    # -------------------------

    # 1. Tải VAE (Đã huấn luyện)
    vae = SimpleVAE(in_channels=3, base_ch=64, z_channels=LATENT_Z_CHANNELS)
    if not os.path.exists(VAE_CHECKPOINT_PATH):
        raise FileNotFoundError(f"VAE checkpoint '{VAE_CHECKPOINT_PATH}' not found. "
                                "Please train VAE first using Latent_Diffusion.py and rename checkpoint.")
    
    try:
        vae_ckpt = torch.load(VAE_CHECKPOINT_PATH, map_location=device)
        if 'vae_state_dict' in vae_ckpt:
            vae.load_state_dict(vae_ckpt['vae_state_dict'])
        else:
            vae.load_state_dict(vae_ckpt)
    except Exception as e:
        raise RuntimeError(f"Error loading VAE checkpoint: {e}")
        
    vae.to(device)
    print("Pre-trained VAE loaded and moved to device.")
    # (Trainer sẽ tự động đóng băng VAE)

    # 2. Khởi tạo UNet cho LATENT SPACE
    model = ConditionalUNet(
        in_channels=LATENT_Z_CHANNELS, # <-- Sửa: Hoạt động trên kênh latent
        base_ch=128,                   # Tăng base_ch vì ảnh latent nhỏ
        ch_mults=(1, 2, 4),
        num_res_blocks=2,
        text_emb_dim=768               # Khớp với CLIP
    )
    model.to(device)

    text_embedder = CLIPTextEmbedder(device=device)

    # 3. Khởi tạo Diffusion cho LATENT SPACE
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        text_embedder=text_embedder,
        image_size=LATENT_IMAGE_SIZE, # <-- Sửa: Kích thước latent
        channels=LATENT_Z_CHANNELS,   # <-- Sửa: Số kênh latent
        timesteps=timesteps,
        device=device
    )
    # (Diffusion đã tối ưu CFG-Train sẽ tự cache null_context)

    # -------------------------
    # Sampler (TỐI ƯU HÓA CFG-SAMPLE)
    # -------------------------
    sampler_cfg = SamplerConfig(steps=50, device=device, guidance_scale=7.0)
    sampler = DDIMSampler(diffusion, sampler_cfg) # Sử dụng sampler đã tối ưu

    # -------------------------
    # Metrics / Logger / Evaluator
    # -------------------------
    metrics = Metrics(device=device)

    # Lấy ảnh thật (pixel) để đánh giá FID
    n_val = min(16, len(image_paths))
    real_images_list = []
    for i in range(n_val):
        img, _ = dataset[i] # Lấy ảnh đã transform
        real_images_list.append(img)
    
    real_images = torch.stack(real_images_list, dim=0) # [-1,1]
    real_images = (real_images + 1.0) / 2.0            # [0,1]
    real_images = real_images.to(device)

    # Đảm bảo prompts khớp với số lượng ảnh
    prompts_for_eval = [captions[i] for i in range(n_val)]
    evaluator = Evaluator(metrics, prompts_for_eval, real_images)

    logger = TrainingLogger(logdir="logs", use_wandb=False)

    # -------------------------
    # Optimizer & Trainer (TỐI ƯU HÓA AMP)
    # -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        vae=vae, # <-- Truyền VAE đã tải vào Trainer
        optimizer=optimizer,
        sampler=sampler,
        dataloader=dataloader,
        evaluator=evaluator,
        logger=logger,
        device=device
    )
    # (Trainer đã tối ưu sẽ xử lý VAE freezing, AMP, LDM logic)

    # -------------------------
    # Training loop
    # -------------------------
    
    ckpt_dir = "checkpoints_quickdraw_ldm" # Đổi tên thư mục checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = os.path.join(ckpt_dir, ckpt_resume_path)
    start_epoch = 0 # Sửa: bắt đầu từ 0
    
    if os.path.exists(resume_path):
        try:
            start_epoch = trainer.load_checkpoint(resume_path)
            print(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Starting from scratch. Error: {e}")
            start_epoch = 0
    else:
        print("Starting training from scratch")
        
    end_epoch = start_epoch + epochs # Sửa: tính toán end_epoch

    # --- Training loop (Logic vòng lặp giữ nguyên) ---
    for epoch in range(start_epoch, end_epoch): # Sửa: range(start, end)
        print(f"=== Epoch {epoch+1}/{end_epoch} ===")
        avg_loss = trainer.train_one_epoch(epoch) # Trainer xử lý mọi logic LDM/AMP

        # Lưu checkpoint
        if (epoch + 1) % 25 == 0 or (epoch + 1) == end_epoch: # Sửa: lưu thường xuyên hơn
            ckpt_path_epoch = os.path.join(ckpt_dir, f"ckpt_epoch{epoch+1}.pt")
            trainer.save_checkpoint(ckpt_path_epoch, epoch=epoch+1)
            print(f'Epoch {epoch+1} Loss: {avg_loss:.5f}')

    # Luôn lưu checkpoint cuối cùng
    ckpt_path_last = os.path.join(ckpt_dir, "ckpt_last.pt")
    trainer.save_checkpoint(ckpt_path_last, epoch=end_epoch)
    print(f'Finished training. Final checkpoint saved to {ckpt_path_last}')


def inference(
    ckpt_path="checkpoints_quickdraw_ldm/ckpt_last.pt", # Sửa: đường dẫn LDM
    prompt="cat",
    out_file="sample.png",
    image_size=32, # Kích thước pixel gốc
    device=None,
):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device} for LDM Inference")

    # -------------------------
    # Model / Diffusion / VAE (TỐI ƯU HÓA LDM)
    # -------------------------

    # 1. Tải VAE (Cần thiết để decode)
    vae = SimpleVAE(in_channels=3, base_ch=64, z_channels=LATENT_Z_CHANNELS).to(device)
    if not os.path.exists(VAE_CHECKPOINT_PATH):
        raise FileNotFoundError(f"VAE checkpoint '{VAE_CHECKPOINT_PATH}' not found.")
    
    try:
        vae_ckpt = torch.load(VAE_CHECKPOINT_PATH, map_location=device)
        if 'vae_state_dict' in vae_ckpt:
            vae.load_state_dict(vae_ckpt['vae_state_dict'])
        else:
            vae.load_state_dict(vae_ckpt)
    except Exception as e:
        raise RuntimeError(f"Error loading VAE checkpoint: {e}")
    vae.eval() # Rất quan trọng
    print("Pre-trained VAE loaded for decoding.")

    # 2. Khởi tạo UNet cho LATENT SPACE
    model = ConditionalUNet(
        in_channels=LATENT_Z_CHANNELS, # <-- Sửa
        base_ch=128,
        ch_mults=(1, 2, 4),
        num_res_blocks=2,
        text_emb_dim=768
    ).to(device)
    
    text_embedder = CLIPTextEmbedder(device=device)

    # 3. Khởi tạo Diffusion cho LATENT SPACE
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        text_embedder=text_embedder,
        image_size=LATENT_IMAGE_SIZE, # <-- Sửa
        channels=LATENT_Z_CHANNELS,    # <-- Sửa
        timesteps=500, # Phải khớp với timesteps lúc train
        device=device
    )

    # 4. Sampler (TỐI ƯU HÓA CFG-SAMPLE)
    sampler_cfg = SamplerConfig(steps=50, device=device, guidance_scale=7.0)
    sampler = DDIMSampler(diffusion, sampler_cfg) # Sampler đã tối ưu

    # -------------------------
    # Load checkpoint
    # -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Dummy optimizer
    
    # Khởi tạo Trainer (dummy) để load checkpoint
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        vae=vae, # Truyền vae
        optimizer=optimizer,
        sampler=sampler,
        dataloader=None,
        evaluator=None,
        logger=None,
        device=device
    )

    if os.path.exists(ckpt_path):
        trainer.load_checkpoint(ckpt_path)
        print(f"Loaded LDM checkpoint from {ckpt_path}")
        
        # --- Benchmark hiệu suất (Giữ nguyên) ---
        print("Running benchmarks...")
        benchmark_text_encoder(text_embedder, batch_size=8, seq_len=77, n_iters=50, device=device)
        # Benchmark UNet
        benchmark_iterations(model, diffusion, batch_size=8, image_size=LATENT_IMAGE_SIZE, n_iters=50, device=device)
        # Benchmark Sampler
        benchmark_images(sampler, trainer.ema_model, text_embedder, batch_size=4, prompt=prompt, n_iters=10, device=device)
    else:
        raise FileNotFoundError(f"No LDM checkpoint found at {ckpt_path}")

    # -------------------------
    # Generate sample (Sửa)
    # -------------------------
    print(f"Generating image for prompt: '{prompt}'...")
    _, context = text_embedder.encode([prompt])
    
    with torch.no_grad():
        # 1. Sampler chạy trong latent space, trả về z_0
        # Sử dụng ema_model để có chất lượng tốt hơn
        z_samples = sampler.sample(batch_size=1, model=trainer.ema_model, context=context)
        
        # 2. Dùng VAE để decode z_0 -> ảnh pixel
        samples = vae.decode(z_samples)

    # [-1,1] → [0,1]
    out = (samples + 1.0) / 2.0
    out = torch.clamp(out, 0.0, 1.0)
    save_image(out, out_file)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    
    # CHỌN CHẾ ĐỘ CHẠY:
    # 1. 'train_vae': Huấn luyện VAE trước.
    # 2. 'train_ldm': Huấn luyện LDM (yêu cầu VAE đã huấn luyện).
    # 3. 'inference': Chạy inference (yêu cầu VAE và LDM đã huấn luyện).
    
    mode = 'train_vae' # <-- THAY ĐỔI CHẾ ĐỘ Ở ĐÂY

    # Lấy device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Đường dẫn dataset LDM của bạn
    my_dataset_path = r'C:\Users\PC\OneDrive\Desktop\scientific research\conditional DDPM\dataset_test\dataset_test'

    if mode == "train_vae":
        print("--- Mode: Training VAE ---")
        # Kiểm tra xem VAE_CHECKPOINT_PATH có tồn tại không
        if os.path.exists(VAE_CHECKPOINT_PATH):
             print(f"Warning: '{VAE_CHECKPOINT_PATH}' đã tồn tại. Sẽ ghi đè nếu huấn luyện.")
             # Thêm logic hỏi người dùng nếu cần
             
        vae_config = VAEConfig(
            epochs=50, # Huấn luyện VAE kỹ
            batch_size=64,
            save_path='./vae_checkpoints', # Lưu checkpoint VAE
            dataset_root=my_dataset_path
        )
        run_vae_training(image_size=32, config=vae_config, device=dev)

    elif mode == "train_ldm":
        print("--- Mode: Training LDM ---")
        if not os.path.exists(VAE_CHECKPOINT_PATH):
            print(f"Error: '{VAE_CHECKPOINT_PATH}' not found.")
            print("Please run mode='train_vae' first.")
        else:
            training(
                root_dir=my_dataset_path,
                ckpt_resume_path='ckpt_last.pt',
                timesteps=500,
                epochs=100, # Tăng epochs cho LDM
                batch_size=16,
                device=dev
            )
            
    elif mode == "inference":
        print("--- Mode: Inference ---")
        if not os.path.exists(VAE_CHECKPOINT_PATH):
             print(f"Error: '{VAE_CHECKPOINT_PATH}' not found. Cannot run inference.")
        else:
            inference(
                ckpt_path=r"checkpoints_quickdraw_ldm\ckpt_last.pt", 
                prompt="apple", 
                out_file="apple_ldm_optimized.png",
                device=dev
            )
    else:
        print(f"Unknown mode: {mode}")