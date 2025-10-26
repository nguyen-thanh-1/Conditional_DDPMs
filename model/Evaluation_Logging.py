# Evaluation_Logging.py
"""
Mục đích:
- (Giữ nguyên) Cung cấp class `Metrics` sử dụng `torchmetrics` (FID, CLIPScore)
  để tính toán hiệu suất.
- (Giữ nguyên) Cung cấp class `TrainingLogger` để ghi log ra TensorBoard/WandB.
- (SỬA ĐỔI) Cung cấp class `Evaluator`:
    - `__init__` được sửa đổi để lưu trữ CẢ (prompts, boxes, labels)
      cho việc đánh giá cố định (fixed evaluation).
    - `evaluate` (Giữ nguyên) so sánh ảnh được sinh ra với ảnh thật.
"""

import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple
import torch.nn as nn

# (Sử dụng phiên bản torchmetrics hiệu suất cao)
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.multimodal.clip_score import CLIPScore
except ImportError:
    print("Lỗi: Vui lòng cài đặt torchmetrics. Chạy: pip install torchmetrics[multimodal]")
    exit()

import torch.nn.functional as F

# -----------------------------------------------------------------
# BƯỚC 1: CLASS METRICS (Sử dụng torchmetrics)
# -----------------------------------------------------------------

class Metrics(nn.Module):
    """
    Tính toán FID và CLIP Score.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Khởi tạo FID
        self.fid = FrechetInceptionDistance(feature=2048).to(device)
        
        # Khởi tạo CLIP Score
        # (Sử dụng model nhanh hơn, ví dụ 'openai/clip-vit-base-patch16')
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    @torch.no_grad()
    def _to_uint8(self, images: torch.Tensor) -> torch.Tensor:
        """Chuyển ảnh từ [0, 1] (float) sang [0, 255] (uint8)."""
        return (images * 255).to(torch.uint8)

    @torch.no_grad()
    def calculate_fid(self, real_images: torch.Tensor, gen_images: torch.Tensor) -> torch.Tensor:
        """Tính FID. Cả hai input đều là [0, 1]."""
        real_uint8 = self._to_uint8(real_images)
        gen_uint8 = self._to_uint8(gen_images)
        
        self.fid.update(real_uint8, real=True)
        self.fid.update(gen_uint8, real=False)
        return self.fid.compute()

    @torch.no_grad()
    def calculate_clip_score(self, gen_images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """Tính CLIP Score. Input ảnh là [0, 1]."""
        gen_uint8 = self._to_uint8(gen_images)
        
        self.clip_score.update(gen_uint8, prompts)
        return self.clip_score.compute()

# -----------------------------------------------------------------
# BƯỚC 2: CLASS LOGGER (TensorBoard)
# -----------------------------------------------------------------

class TrainingLogger:
    """Ghi log ra TensorBoard."""
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int, normalize: bool = True):
        grid = make_grid(images, normalize=normalize, value_range=(0, 1))
        self.writer.add_image(tag, grid, step)
        
    def log_fid(self, fid_score: float, step: int):
        self.log_scalar("metrics/FID", fid_score, step)

    def log_clip_score(self, clip_score: float, step: int):
        self.log_scalar("metrics/CLIP_Score", clip_score, step)

    def close(self):
        self.writer.close()

# -----------------------------------------------------------------
# BƯỚC 3: CLASS EVALUATOR (ĐÃ SỬA ĐỔI)
# -----------------------------------------------------------------

class Evaluator:
    """
    Lớp này lưu trữ một batch đánh giá cố định (fixed) 
    và tính toán metrics khi được yêu cầu.
    """
    def __init__(self, 
                 metrics: Metrics, 
                 prompts: List[str], 
                 real_images: torch.Tensor,
                 # --- MỚI: Thêm điều kiện spatial ---
                 boxes: List[torch.Tensor],   
                 labels: List[List[str]]   
                 # --- (Kết thúc) ---
                ):
        self.metrics = metrics
        
        # --- MỚI: Lưu trữ TẤT CẢ điều kiện ---
        # (Trainer (Module 2) sẽ đọc chúng từ đây để sinh ảnh eval)
        self.prompts = prompts
        self.boxes = boxes
        self.labels = labels
        # --- (Kết thúc) ---
        
        # Đảm bảo real_images (ảnh thật) ở [0, 1]
        self.real_images = real_images.clamp(0.0, 1.0) 

    @torch.no_grad()
    def evaluate(self, gen_images: torch.Tensor) -> Tuple[float, float]:
        """
        Hàm này (Giữ nguyên).
        Nó chỉ so sánh KẾT QUẢ (gen_images)
        với DỮ LIỆU THẬT (real_images, prompts).
        """
        
        # Đảm bảo gen_images (ảnh được sinh ra) ở [0, 1]
        gen_images_clamped = gen_images.clamp(0.0, 1.0)
        
        # 1. Tính FID (so sánh ảnh-với-ảnh)
        fid = self.metrics.calculate_fid(self.real_images, gen_images_clamped).item()
        
        # 2. Tính CLIP Score (so sánh ảnh-với-text)
        clip_score = self.metrics.calculate_clip_score(gen_images_clamped, self.prompts).item()
        
        return fid, clip_score