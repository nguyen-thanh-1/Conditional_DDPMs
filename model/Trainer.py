# Trainer.py
"""
Mục đích:
1. Định nghĩa lớp `Trainer` (ĐÃ SỬA ĐỔI cho ControlNet):
   - `__init__`: Nhận `model` (UNet đóng băng) và `control_net` (huấn luyện).
   - EMA giờ đây áp dụng cho `control_net`.
   - `train_one_epoch`: `loss_fn` được gọi như cũ (logic trừu tượng).
   - Logic Sampling/Evaluation: Đã VIẾT LẠI HOÀN TOÀN để gọi ControlNet.
   - `save/load_checkpoint`: Đã VIẾT LẠI HOÀN TOÀN để chỉ lưu/tải `control_net`.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple

# AMP (Automatic Mixed Precision)
from torch.amp import GradScaler, autocast

try:
    from torch_ema import ExponentialMovingAverage
    from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
except ImportError:
    print("Lỗi: Vui lòng cài đặt thư viện. Chạy: pip install torch-ema diffusers")
    exit()

# Imports từ các module của chúng ta
from Evaluation_Logging import TrainingLogger, Evaluator, Metrics
from Conditioning import ConditionalGaussianDiffusion # Sửa đổi quan trọng

# -----------------------------------------------------------------
# (TÔI ĐÃ XÓA LỚP DATASET Ở ĐÂY - Chúng ta sẽ dùng data_parser_module2.py)
# -----------------------------------------------------------------

class Trainer:
    """
    Lớp Trainer (Đã sửa đổi cho ControlNet).
    Quản lý vòng lặp huấn luyện, EMA, checkpointing, và đánh giá.
    """
    def __init__(
        self,
        diffusion: ConditionalGaussianDiffusion, # <-- Lớp diffusion đã sửa đổi
        dataloader: DataLoader,
        evaluator: Evaluator,
        logger: TrainingLogger,
        optimizer: torch.optim.Optimizer,
        scheduler: SchedulerMixin, # Dùng cho inference
        lr: float = 1e-4,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        save_every_n_steps: int = 1000,
        eval_every_n_steps: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_name: str = "controlnet.pt",
    ):
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.save_every_n_steps = save_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        
        self.device = diffusion.device
        
        # --- THAY ĐỔI: Chỉ theo dõi UNet (model) và ControlNet ---
        self.model = diffusion.model.to(self.device)
        self.control_net = diffusion.control_net.to(self.device)
        # --- (Kết thúc) ---
        
        # VAE và TextEmbedder/SpatialEmbedder không cần huấn luyện
        self.vae = diffusion.vae.to(self.device)
        self.text_embedder = diffusion.text_embedder.to(self.device)
        self.spatial_embedder = diffusion.spatial_embedder.to(self.device)
        
        # --- THAY ĐỔI: EMA áp dụng cho ControlNet (không phải UNet) ---
        self.ema = ExponentialMovingAverage(self.control_net.parameters(), decay=0.995)
        
        # AMP Scaler
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.global_step = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, epochs: int):
        """Vòng lặp huấn luyện chính."""
        start_epoch = self.global_step // len(self.dataloader)
        print(f"Bắt đầu huấn luyện từ epoch {start_epoch+1} / {epochs}")
        
        for epoch in range(start_epoch, epochs):
            print(f"--- Bắt đầu Epoch {epoch+1}/{epochs} ---")
            
            # --- Train 1 epoch ---
            avg_loss = self.train_one_epoch(epoch)
            self.logger.log_scalar("train/loss", avg_loss, epoch)
            
            # --- Lưu Checkpoint cuối cùng của epoch ---
            self.save_checkpoint(
                path=os.path.join(self.checkpoint_dir, f"{self.checkpoint_name}_epoch{epoch+1}.pt"), 
                epoch=epoch
            )
            
        print("--- Huấn luyện hoàn tất ---")
        self.logger.close()

    def train_one_epoch(self, epoch: int) -> float:
        """Logic cho 1 epoch huấn luyện."""
        
        # --- THAY ĐỔI: Đặt ControlNet ở chế độ train(), UNet ở eval() ---
        self.control_net.train()
        self.model.eval() # UNet luôn ở chế độ eval()
        self.text_embedder.eval() # Text embedder luôn ở eval()
        self.vae.eval() # VAE luôn ở eval()
        # --- (Kết thúc) ---
        
        total_loss = 0.0
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        
        for i, batch in enumerate(progress_bar):
            # Lấy dữ liệu (từ collate_fn)
            images = batch["images"].to(self.device) # (B, 3, H, W)
            prompts = batch["prompts"]               # List[str]
            boxes = batch["boxes"]                   # List[Tensor(N, 4)]
            labels = batch["labels"]                 # List[List[str]]
            
            # (Ảnh cần được chuẩn hóa [-1, 1] cho VAE)
            images = images * 2.0 - 1.0
            

            with autocast(self.device.type, enabled=self.use_amp):
                # --- THAY ĐỔI: Truyền tất cả điều kiện vào loss_fn ---
                loss = self.diffusion.loss_fn(images, prompts, boxes, labels)
                loss = loss / self.grad_accum_steps
                
            # Backward
            self.scaler.scale(loss).backward()
            
            # Gradient Accumulation
            if (i + 1) % self.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # --- THAY ĐỔI: EMA cập nhật ControlNet ---
                self.ema.update(self.control_net.parameters())
                
                self.global_step += 1
                progress_bar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})
                self.logger.log_scalar("step/loss", loss.item() * self.grad_accum_steps, self.global_step)

            total_loss += loss.item() * self.grad_accum_steps

            # --- Đánh giá (Evaluation) giữa epoch ---
            if self.global_step % self.eval_every_n_steps == 0:
                self.run_evaluation(step=self.global_step)
            
            # --- Lưu Checkpoint giữa epoch ---
            if self.global_step % self.save_every_n_steps == 0:
                self.save_checkpoint(
                    path=os.path.join(self.checkpoint_dir, f"{self.checkpoint_name}_step{self.global_step}.pt"), 
                    epoch=epoch
                )

        avg_loss = total_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0.0
        print(f"Epoch {epoch+1} trung bình loss: {avg_loss:.5f}")
        return avg_loss

    # --- Hàm save/load checkpoint (VIẾT LẠI) ---
    def save_checkpoint(self, path: str, epoch: int):
        """Lưu checkpoint (CHỈ LƯU CONTROLNET)."""
        torch.save({
            "control_net": self.control_net.state_dict(),
            "spatial_embedder": self.spatial_embedder.state_dict(), 
            "ema": self.ema.state_dict(), 
            "opt": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.global_step,
            "epoch": epoch
        }, path)
        print(f"Checkpoint ControlNet đã lưu tại {path} (Epoch {epoch}, Bước {self.global_step})")

    def load_checkpoint(self, path: str) -> int:
        """Tải checkpoint (CHỈ TẢI CONTROLNET)."""
        if not os.path.exists(path):
            print(f"Cảnh báo: Không tìm thấy checkpoint tại {path}. Bắt đầu huấn luyện từ đầu.")
            return 0
            
        ckpt = torch.load(path, map_location=self.device)
        
        self.control_net.load_state_dict(ckpt["control_net"]) # <-- Tải vào ControlNet
        if "spatial_embedder" in ckpt: # <-- THÊM KHỐI LỆNH NÀY
            self.spatial_embedder.load_state_dict(ckpt["spatial_embedder"])
        else:
            print("Cảnh báo: Không tìm thấy 'spatial_embedder' trong checkpoint.")
        self.ema.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["opt"])
        
        if "scaler" in ckpt and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler"])
            
        self.global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", -1) + 1 # Bắt đầu từ epoch tiếp theo
        
        print(f"Đã tải checkpoint ControlNet từ {path}. Tiếp tục từ Epoch {start_epoch} (Bước {self.global_step})")
        return start_epoch

    @torch.no_grad()
    def run_evaluation(self, step: int):
        """
        Chạy pipeline sinh ảnh (sampling) và tính toán metrics.
        """
        print(f"\n--- Đang chạy đánh giá tại Bước {step} ---")
        
        # --- THAY ĐỔI: Đặt ControlNet vào eval() ---
        self.control_net.eval()
        self.model.eval()
        # --- (Kết thúc) ---

        # Sử dụng EMA weights của ControlNet để sinh ảnh
        with self.ema.average_parameters():
            # --- THAY ĐỔI: Gọi hàm sample (sẽ được định nghĩa) ---
            gen_images = self.sample(
                prompts=self.evaluator.prompts,
                spatial_boxes=self.evaluator.boxes,
                spatial_labels=self.evaluator.labels,
                cfg_scale_text=7.5,
                cfg_scale_spatial=1.0, # (Thường giữ ở 1.0)
                num_steps=50 # (Sử dụng sampler nhanh)
            ) # Output là [0, 1]

        # 1. Tính Metrics (FID, CLIP Score)
        fid, clip_score = self.evaluator.evaluate(gen_images)
        print(f"Đánh giá Bước {step}: FID = {fid:.4f}, CLIP Score = {clip_score:.4f}")
        
        # 2. Log Metrics
        self.logger.log_fid(fid, step)
        self.logger.log_clip_score(clip_score, step)
        
        # 3. Log Ảnh (so sánh thật và giả)
        # (Ảnh thật đã ở [0, 1])
        comparison_grid = torch.cat([self.evaluator.real_images.cpu(), gen_images.cpu()], dim=0)
        self.logger.log_images(
            "eval/Real_vs_Generated", 
            comparison_grid, 
            step,
            normalize=False # Ảnh đã ở [0, 1]
        )
        print("--- Đánh giá hoàn tất ---")
        
        # Trả ControlNet về chế độ train()
        self.control_net.train()

    @torch.no_grad()
    def sample(
        self,
        prompts: List[str],
        spatial_boxes: List[torch.Tensor],
        spatial_labels: List[List[str]],
        cfg_scale_text: float = 7.5,
        cfg_scale_spatial: float = 1.0,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Hàm sinh ảnh (sampling) sử dụng ControlNet và CFG.
        """
        B = len(prompts)
        device = self.device
        
        # --- 1. Thiết lập Scheduler (ví dụ: DPMSolver) ---
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps
        
        # --- 2. Tạo Latent Noise ban đầu ---
        latent_size_H = self.diffusion.latent_size[0]
        latent_size_W = self.diffusion.latent_size[1]
        latents = torch.randn((B, 4, latent_size_H, latent_size_W), device=device)
        latents = latents * self.scheduler.init_noise_sigma
        
        # --- 3. Mã hóa điều kiện (Text & Spatial) ---
        
        # 3a. Điều kiện Positive (có text, có spatial)
        pos_text_list = [self.text_embedder([p]) for p in prompts]
        pos_text = torch.cat(pos_text_list, dim=0)
        
        pos_spatial_list = [self.spatial_embedder(b.to(device), l, device) for b, l in zip(spatial_boxes, spatial_labels)]
        pos_spatial = torch.cat(pos_spatial_list, dim=0)

        # 3b. Điều kiện Negative (Text)
        neg_text = self.diffusion.null_text_context.repeat(B, 1, 1)
        
        # 3c. Điều kiện Negative (Spatial)
        neg_spatial = self.diffusion.null_spatial_context_map.repeat(B, 1, 1, 1)

        # --- 4. Vòng lặp Denoising ---
        for t in tqdm(timesteps, desc="Sampling"):
            # Mở rộng latents cho CFG (input * 2: 1 neg, 1 pos)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Mở rộng các điều kiện
            # (neg_text, pos_text)
            text_input = torch.cat([neg_text, pos_text])
            # (neg_spatial, pos_spatial)
            spatial_input = torch.cat([neg_spatial, pos_spatial])

            # --- Chạy ControlNet (cho cả neg và pos) ---
            control_residuals = self.control_net(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_input,
                controlnet_cond=spatial_input
            )
            
            # --- Chạy UNet (cho cả neg và pos) ---
            noise_pred = self.model(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_input,
                down_block_residual_samples=control_residuals.down_block_res_samples,
                mid_block_residual_sample=control_residuals.mid_block_res_sample
            ).sample
            
            # --- 5. Áp dụng Classifier-Free Guidance (CFG) ---
            noise_pred_neg, noise_pred_pos = noise_pred.chunk(2)
            
            # CFG cho Text
            noise_pred = noise_pred_neg + cfg_scale_text * (noise_pred_pos - noise_pred_neg)
            
            # (Hiện tại chúng ta không dùng CFG cho spatial, nhưng có thể thêm sau)
            
            # --- 6. Cập nhật Latent (scheduler step) ---
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # --- 7. Giải mã Latents -> Ảnh (Pixel) ---
        # (Scale latents trước khi decode)
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        
        # Chuyển từ [-1, 1] sang [0, 1]
        images = (images / 2.0 + 0.5).clamp(0.0, 1.0)
        return images