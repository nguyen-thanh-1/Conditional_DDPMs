# Trainer.py (Phiên bản tối ưu tuyệt đối với Latent Caching)
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from PIL import Image
from torch.cuda.amp import GradScaler, autocast

try:
    from torch_ema import ExponentialMovingAverage
    from diffusers.models import AutoencoderKL
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
except ImportError:
    print("Lỗi: Vui lòng cài đặt thư viện. Chạy: pip install torch-ema diffusers")
    exit()

from Conditioning import ConditionalGaussianDiffusion
from Evaluation_Logging import TrainingLogger, Evaluator, Metrics

# --- Dataset GỐC (Giữ nguyên) ---
# (Chúng ta cần giữ lại class này để chạy precaching)
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, captions, transform=None, tokenizer=None):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, cap = self.image_paths[idx], self.captions[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Using a dummy image. Error: {e}")
            img = Image.new('RGB', (32, 32), (0, 0, 0)) # Đảm bảo ảnh giả có kích thước
        if self.transform:
            img = self.transform(img)
        return img, cap

# --- TỐI ƯU HÓA: Dataset MỚI (Đọc Latent đã cache) ---
class LatentCaptionDataset(torch.utils.data.Dataset):
    """
    Dataset này đọc các latents đã được VAE encode và cache trước.
    Nó chỉ load tensor từ file .pt và caption tương ứng.
    """
    def __init__(self, cache_dir: str, captions: list):
        self.cache_dir = cache_dir
        self.captions = captions
        # Giả định file cache được đặt tên là latent_0.pt, latent_1.pt, ...
        self.num_files = len(captions)
        
        # Kiểm tra nhanh
        if self.num_files == 0:
            raise ValueError(f"No captions provided for LatentCaptionDataset.")
        expected_file = os.path.join(self.cache_dir, f"latent_{self.num_files - 1}.pt")
        if not os.path.exists(expected_file):
            print(f"Warning: Cache dir '{cache_dir}' or file '{expected_file}' might be missing.")
            print("Ensure precaching was run successfully.")
            
    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        latent_path = os.path.join(self.cache_dir, f"latent_{idx}.pt")
        
        try:
            # Load latent (đã được scale) từ đĩa
            # Dùng map_location='cpu' để tránh tăng VRAM khi load
            latent = torch.load(latent_path, map_location='cpu') 
        except Exception as e:
            print(f"Error loading latent {latent_path}. {e}")
            # Trả về tensor rỗng để training có thể skip (nếu có check NaN)
            latent = torch.zeros(4, 4, 4) # Cần khớp shape (C, H, W)
            
        caption = self.captions[idx]
        return latent, caption
# --- KẾT THÚC TỐI ƯU HÓA ---


# --- Trainer (TỐI ƯU HÓA) ---
class Trainer:
    # (Init giữ nguyên)
    def __init__(self, 
                 model: torch.nn.Module, 
                 diffusion: ConditionalGaussianDiffusion, 
                 vae: AutoencoderKL, # <-- VAE VẪN CẦN cho evaluation
                 optimizer: torch.optim.Optimizer, 
                 scheduler: SchedulerMixin,
                 dataloader: DataLoader, # <-- DataLoader này giờ là của LatentCaptionDataset
                 evaluator: Evaluator, 
                 logger: TrainingLogger,
                 device: torch.device,
                 vae_scaling_factor: float, # <-- Vẫn cần cho evaluation decode
                 ema_decay: float = 0.999,
                 sampling_steps: int = 50,
                 guidance_scale: float = 7.0
                 ):
        
        self.model = model.to(device)
        self.diffusion = diffusion
        self.vae = vae.to(device) # VAE giờ chỉ dùng cho evaluation
        self.vae_scaling_factor = vae_scaling_factor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.logger = logger
        self.device = device

        self.sampling_steps = sampling_steps
        self.guidance_scale = guidance_scale

        # VAE đã được đóng băng bên ngoài
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # EMA (torch_ema)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        
        # --- SỬA CẢNH BÁO: Cập nhật cú pháp GradScaler ---
        self.use_amp = (self.device.type == 'cuda')
        device_type = 'cuda' if self.use_amp else 'cpu'
        self.scaler = torch.amp.GradScaler(device=device_type, enabled=self.use_amp)
        print(f"Trainer initialized with device: {device} | AMP Enabled: {self.use_amp}")
        # --- KẾT THÚC SỬA CẢNH BÁO ---

        self.global_step = 0
        self.null_context = self.diffusion.null_context.to(device)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        
        for i, batch in enumerate(self.dataloader):
            # --- TỐI ƯU HÓA: Batch giờ là (latents, captions) ---
            # (Không còn 'imgs' nữa)
            z, caps_device = batch
            z = z.to(self.device) # Chuyển latent (đã scale) sang device
            # caps_device là list captions (giữ nguyên)
            # --- KẾT THÚC TỐI ƯU HÓA ---

            self.optimizer.zero_grad()
            
            device_type = 'cuda' if self.use_amp else 'cpu'
            with torch.autocast(device_type=device_type, enabled=self.use_amp):
                
                # --- TỐI ƯU HÓA: XÓA BỎ HOÀN TOÀN VAE ENCODE ---
                # (Block 'vae.encode(imgs)' đã bị xóa)
                # --- KẾT THÚC TỐI ƯU HÓA ---
                
                # 4. Tính loss trực tiếp trên latent z
                loss = self.diffusion.loss_fn(z, caps_device) 

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at step {self.global_step}. Skipping batch.")
                self.optimizer.zero_grad()
                continue
                
            # Backward (với AMP)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Cập nhật EMA
            self.ema.update()

            total_loss += loss.item()

            if self.global_step % 50 == 0:
                 self.logger.log_losses({"train_loss": loss.item()}, self.global_step)

            # --- Logic Sampling/Evaluation (Giữ nguyên) ---
            # (Phần này không thay đổi vì VAE vẫn được lưu trong self.vae
            #  và dùng để decode khi evaluation)
            if self.global_step % 1000 == 0 and self.evaluator is not None:
                # ... (Toàn bộ logic evaluation giữ nguyên) ...
                # (Vẫn dùng self.vae.decode() ở cuối)
                # print(f"Step {self.global_step}: Logging samples and evaluating...")
                
                with self.ema.average_parameters():
                    self.model.eval()
                    with torch.no_grad():
                        sample_captions = list(caps_device[:4]) # Lấy caption từ batch
                        if not sample_captions:
                            continue
                            
                        B = len(sample_captions)
                        
                        # 1. Encode context
                        _, context = self.diffusion.text_embedder.encode(sample_captions)
                        context = context.to(self.device)
                        null_context = self.null_context.repeat(B, 1, 1)

                        # 2. Chuẩn bị scheduler
                        self.scheduler.set_timesteps(self.sampling_steps)
                        
                        # 3. Khởi tạo latent
                        z_shape = (B, self.diffusion.channels, self.diffusion.image_size, self.diffusion.image_size)
                        latents = torch.randn(z_shape, device=self.device)
                        
                        # 4. Vòng lặp sampling
                        device_type = 'cuda' if self.use_amp else 'cpu'
                        with torch.autocast(device_type=device_type, enabled=self.use_amp):
                            for t in self.scheduler.timesteps:
                                latent_model_input = torch.cat([latents] * 2)
                                t_in = t.repeat(B * 2).to(self.device)
                                context_in = torch.cat([null_context, context])

                                noise_pred_all = self.model(
                                    sample=latent_model_input, 
                                    timestep=t_in, 
                                    encoder_hidden_states=context_in
                                ).sample
                                
                                noise_pred_uncond, noise_pred_cond = noise_pred_all.chunk(2)
                                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                        # 5. Decode VAE (Giữ nguyên)
                        latents_unscaled = latents / self.vae_scaling_factor
                        samples_pixel = self.vae.decode(latents_unscaled).sample
                        
                        # 6. Log
                        samples_pixel_log = (samples_pixel + 1.0) / 2.0
                        self.logger.log_images(samples_pixel_log, self.global_step, name="generated_samples")
                        
                        # 7. Đánh giá
                        results = self.evaluator.evaluate(samples_pixel_log)
                        self.logger.log_metrics(results, self.global_step)
                
                self.model.train() 
            
            self.global_step += 1
            
        avg_loss = total_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0.0
        print(f"Epoch {epoch} average loss: {avg_loss:.5f}")
        return avg_loss

    # --- Hàm save/load checkpoint (Giữ nguyên) ---
    def save_checkpoint(self, path: str, epoch: int):
        # (Không thay đổi)
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(), 
            "opt": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.global_step,
            "epoch": epoch
        }, path)
        print(f"Checkpoint saved to {path} (Epoch {epoch}, Step {self.global_step})")

    def load_checkpoint(self, path: str) -> int:
        # (Không thay đổi)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ema.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt and self.use_amp:
             self.scaler.load_state_dict(ckpt["scaler"])
             
        self.global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        
        print(f"Loaded checkpoint from {path}, resume at epoch {start_epoch} (step {self.global_step})")
        return start_epoch