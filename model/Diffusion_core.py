# Diffusion_core.py
"""
Mục đích:
- (Tối ưu hóa) Cung cấp logic training cốt lõi của Gaussian Diffusion 
  (tính schedule, q_sample).
- (Không đổi) Đây là module nền tảng toán học cho diffusion.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Utilities: Noise Schedules --------------------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Linear schedule from beta_start to beta_end over timesteps."""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (improved)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# -------------------- Gaussian Diffusion Core --------------------

class GaussianDiffusion(nn.Module):
    """
    Lớp Gaussian Diffusion cơ sở.
    Chỉ chứa logic toán học cho noising (q_sample) và schedule.
    Logic conditional (loss_fn, v.v.) sẽ ở lớp con.
    """
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()
        self.timesteps = timesteps

        # Xác định schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.register_buffer("betas", betas)

        # Tính toán các hằng số alpha
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # --- Các hằng số cần thiết cho q_sample (noising) ---
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # --- Các hằng số cần thiết cho q_posterior (denoising - ít dùng khi train) ---
        self.register_buffer('posterior_variance', betas * (1. - self.alphas_cumprod_prev) / (1. - alphas_cumprod))

   # (Code mới đã sửa lỗi)
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) (forward process / noising).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # --- SỬA LỖI DEVICE MISMATCH ---
        
        # 1. Lấy device của buffer (là 'cpu') và device của x_start (là 'cuda')
        buffer_device = self.sqrt_alphas_cumprod.device
        x_device = x_start.device
        
        # 2. Chuyển 't' về device của buffer ('cpu') để index
        t_on_buffer_device = t.to(buffer_device)
        
        # 3. Index trên 'cpu'
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_on_buffer_device].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_on_buffer_device].view(-1, 1, 1, 1)
        
        # 4. Chuyển các hằng số vừa lấy ra ('cpu') lên device của x_start ('cuda')
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.to(x_device)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.to(x_device)
        
        # --- KẾT THÚC SỬA LỖI ---
        
        # Công thức q_sample (bây giờ tất cả đều trên 'cuda')
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def loss_fn(self, x_start: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Hàm loss_fn gốc (unconditional).
        Lớp ConditionalGaussianDiffusion sẽ GHI ĐÈ (override) hàm này.
        """
        B, C, H, W = x_start.shape
        device = x_start.device
        
        # 1. Sample t
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        
        # 2. Sample noise
        noise = torch.randn_like(x_start)
        
        # 3. Noise image (q_sample)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 4. Predict noise
        predicted_noise = self.model(x_t, t) # 'self.model' (UNet) phải được định nghĩa ở lớp con
        
        # 5. Calculate loss (thường là L1 hoặc L2)
        loss = F.mse_loss(noise, predicted_noise)
        return loss