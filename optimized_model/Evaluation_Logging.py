# Evaluation_Logging.py (Phiên bản tối ưu với torchmetrics)
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

# --- THAY ĐỔI: Imports thư viện metrics ---
# Cài đặt: pip install torchmetrics[multimodal]
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.multimodal.clip_score import CLIPScore
except ImportError:
    print("Lỗi: Vui lòng cài đặt torchmetrics. Chạy: pip install torchmetrics[multimodal]")
    exit()
# --- KẾT THÚC THAY ĐỔI ---

import torch.nn.functional as F

class Metrics:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        
        # --- THAY ĐỔI: Sử dụng torchmetrics ---
        print("Initializing TorchMetrics (FID, CLIPScore)...")
        # 1. Khởi tạo FrechetInceptionDistance (FID)
        # feature=2048 dùng pool3 (InceptionV3)
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
        
        # 2. Khởi tạo CLIPScore
        # Tự động tải model
        self.clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(self.device)
        print("TorchMetrics initialized.")
        # --- KẾT THÚC THAY ĐỔI ---

    def _to_uint8(self, images: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi tensor ảnh từ [0, 1] (float) sang [0, 255] (uint8)."""
        # Đảm bảo ảnh nằm trong khoảng [0, 1] trước khi nhân
        images = torch.clamp(images, 0.0, 1.0)
        return (images * 255).to(torch.uint8)

    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Tính FID bằng torchmetrics.
        Input: Tensors [0, 1] (float)
        """
        # Chuyển đổi ảnh sang đúng định dạng (uint8) và device
        real_images_uint8 = self._to_uint8(real_images.to(self.device))
        fake_images_uint8 = self._to_uint8(fake_images.to(self.device))

        # 2. Cập nhật metric
        self.fid_metric.update(real_images_uint8, real=True)
        self.fid_metric.update(fake_images_uint8, real=False)
        
        # 3. Tính toán và trả về
        fid_value = self.fid_metric.compute().item()
        self.fid_metric.reset() # Reset để lần gọi sau không bị ảnh hưởng
        return fid_value

    def calculate_clip_score(self, images: torch.Tensor, prompts: list[str]) -> float:
        """
        Tính CLIP Score bằng torchmetrics.
        Input: Tensors [0, 1] (float)
        """
        images_uint8 = self._to_uint8(images.to(self.device))

        # 2. Tính toán (Nội bộ torchmetrics sẽ xử lý preprocessing)
        self.clip_score_metric.update(images_uint8, prompts)
        
        # 3. Trả về
        clip_value = self.clip_score_metric.compute().item()
        self.clip_score_metric.reset()
        return clip_value


# Logger (Giữ nguyên)
class TrainingLogger:
    def __init__(self, logdir="logs", use_wandb=False, project="ddpm-training"):
        self.writer = SummaryWriter(logdir)
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project)
                print(f"WandB logging enabled for project '{project}'.")
            except ImportError:
                print("WandB not installed. Disabling WandB logging.")
                self.use_wandb = False

    def log_losses(self, losses: dict, step: int):
        for k, v in losses.items():
            self.writer.add_scalar(f"loss/{k}", v, step)
            if self.use_wandb:
                import wandb
                wandb.log({f"loss/{k}": v, "step": step})

    def log_images(self, images: torch.Tensor, step: int, name="samples"):
        # Giả định images đang ở [0, 1]
        grid = make_grid(images, nrow=4, normalize=False) # Đã ở [0,1]
        self.writer.add_image(name, grid, step)
        if self.use_wandb:
            import wandb
            # Cần permute về (H, W, C) cho wandb
            grid_wandb = grid.permute(1, 2, 0).cpu().numpy()
            wandb.log({name: [wandb.Image(grid_wandb)], "step": step})

    def log_metrics(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"metric/{k}", v, step)
            if self.use_wandb:
                import wandb
                wandb.log({f"metric/{k}": v, "step": step})


# Evaluator (Giữ nguyên)
# Nó vẫn hoạt động vì class Metrics giữ nguyên API
class Evaluator:
    def __init__(self, metrics: Metrics, prompts: list[str], real_images: torch.Tensor):
        self.metrics = metrics
        self.prompts = prompts
        # Đảm bảo real_images ở [0, 1]
        self.real_images = real_images.clamp(0.0, 1.0) 

    def evaluate(self, gen_images: torch.Tensor):
        # Đảm bảo gen_images ở [0, 1]
        gen_images_clamped = gen_images.clamp(0.0, 1.0)
        
        fid = self.metrics.calculate_fid(self.real_images, gen_images_clamped)
        clip_score = self.metrics.calculate_clip_score(gen_images_clamped, self.prompts)
        return {"FID": fid, "CLIPScore": clip_score}