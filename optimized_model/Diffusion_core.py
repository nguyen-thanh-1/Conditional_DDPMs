"""
Purpose:
- (Tối ưu hóa) Cung cấp logic training cốt lõi của Gaussian Diffusion (tính schedule, q_sample và loss_fn).
- (Tối ưu hóa) Kiến trúc UNet và logic Sampling đã được chuyển sang các thư viện 
  'diffusers.models.UNet2DConditionModel' và 'diffusers.schedulers'
  và sẽ được gọi từ các file cấp cao hơn.
"""

from typing import Optional, Tuple
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Utilities: Noise Schedules --------------------
# (Giữ nguyên - Đây là logic toán học cơ bản, nhẹ và chính xác)

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Linear schedule from beta_start to beta_end over timesteps."""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (improved).
    Returns beta_t for t in 0..timesteps-1 as a torch.Tensor.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)

# -------------------- Toàn bộ UNet, Blocks... đã bị XÓA --------------------
# (Sẽ được thay thế bằng diffusers.models.UNet2DConditionModel)


# -------------------- Gaussian Diffusion (TỐI ƯU HÓA) --------------------

@dataclass
class DiffusionHyperparams:
    timesteps: int = 1000
    beta_schedule: str = 'linear'  # 'linear' or 'cosine'

class GaussianDiffusion:
    """
    Lớp Gaussian Diffusion đã được tối ưu:
    - Chỉ chứa logic cho training (tính schedule, q_sample, loss_fn).
    - Toàn bộ logic sampling (p_sample, sample,...) đã bị XÓA BỎ.
    - Việc sampling sẽ được xử lý bên ngoài bởi thư viện (vd: diffusers.schedulers).
    """
    def __init__(self, model: nn.Module, image_size: int, channels: int,
                 timesteps: int, beta_schedule: str = 'linear', 
                 device: Optional[torch.device] = None):
        
        # Lưu ý: model giờ đây là một model BẤT KỲ (vd: UNet2DConditionModel)
        # tuân thủ signature model(x, t, context=None)
        self.model = model 
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unknown beta schedule')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # register as buffers
        self.register_buffer = lambda name, val: setattr(self, name, val.to(self.device))

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # calculations for q_sample (forward process)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) (forward process).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def loss_fn(self, x_start: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss L_simple (MSE between true noise and predicted noise).
        (Phiên bản này dành cho unconditional DDPM)
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)
        
        # Giả định model(x, t, context=None)
        eps_pred = self.model(x_t, t, context=None) 
        
        return F.mse_loss(eps_pred, noise)

    # --- TOÀN BỘ CÁC HÀM SAMPLING ĐÃ BỊ XÓA ---