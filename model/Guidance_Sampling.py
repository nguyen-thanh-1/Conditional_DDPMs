"""
Purpose:
- Implement fast and flexible samplers for trained diffusion models:
  - DDIM deterministic sampler (non-Markovian) for fewer steps
  - PNDM/PLMS-style multistep sampler skeleton (improved sampling quality)
  - Batch-efficient classifier-free guidance implementation (single forward pass trick)
- Utilities for EMA weights, mixed-precision inference, and sampling config
- Example inference script showing common usage patterns

Notes:
- This module expects a model implementing the signature `model(x, t, context=None)` where
  `context` is either `None` or a tensor of shape (B, L, D). The model should return predicted noise `eps`.
- For classifier-free guidance we use the common concatenation trick: pass a batch where the first half
  are unconditioned inputs and the second half conditioned, then combine.
"""

from typing import Optional, Tuple, List, Dict
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.cuda.amp import autocast # <-- Thêm

# import module1 and module2 utilities (assumed to be in same package)
from Conditioning import ConditionalGaussianDiffusion


# -------------------- EMA utility --------------------
'''
Goal: Theo dõi Exponential Moving Average của các tham số trainable của model và có thể thay các tham số hiện tại bằng giá trị EMA để inference/eval, rồi khôi phục lại tham số gốc
    lưu và áp dụng trọng số trung bình trượt để inference ổn định 
'''
class EMA: # Exponential Moving Average

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.model = model
        self.decay = decay # hệ số EMA (scalar), 0 <= decay <= 1
        self.device = device
        self.shadow = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad} # Bản sao của các tham số trainable, lưu giá trị EMA
    def update(self):
        for name, p in self.model.named_parameters():
            if not p.requires_grad: # chỉ láy những tham số mà được tối ưu 
                continue
            self.shadow[name] = (self.decay * self.shadow[name].to(p.device) + (1.0 - self.decay) * p.detach().cpu()).cpu()
            # cập nhật self.shadow: trộn giá trọ EMA cũ với giá trị tham số hiện tại của model
    def apply_shadow(self):
        self._backup = {name: p.detach().cpu().clone() for name, p in self.model.named_parameters() if p.requires_grad} # dict tạm: lưu snapshot tham số gốc để restore sau khi apply
        for name, p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            p.data.copy_(self.shadow[name].to(p.device))
        # model parameter p chứa giá trị EMA (phù hợp cho inference/eval)

    def restore(self):
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self._backup[name].to(p.device))
        del self._backup

'''
Giữ 'shadow' weight (EMA của trọng số) để dùng lúc inference (thường cho kết quả ổn định)

update(): cập nhật shadow từ params hiện tại theo decay
apply_shadow(): copy shadow -> model (backup hiện tại vào _backup)
restore(): khôi phục tham số cũ
'''
# -------------------- Sampling Schedulers --------------------
# lưu các tham số sampling của mô hình để truyền vào sampler như DDIPSampler or PNDMSampler
@dataclass
class SamplerConfig: # nơi điều chỉnh thử nghiệm quá trình sinh ảnh
    sampler: str = 'ddim'  # 'ddim' or 'pndm' (pndm is skeleton)
    steps: int = 50 # phân hạch thời gian | xác định số bước láy mẫu 
    eta: float = 0.0  # for DDIM: 0.0 deterministic, 1.0 stochastic | điều chỉnh mức độ ngẫu nhiên | quyết định xem có thêm nhiễu z (stochastic sampling) hay không
    guidance_scale: float = 4.0 # hệ số trong Classifier-Free Guidance (CFG) - điều chỉnh sức ảnh hưởng của điện kiện (context) 
    device: Optional[torch.device] = None



class DDIMSampler:
    """DDIM sampler implementation that can sample with fewer steps using non-Markovian updates.
    Formula reference: DDIM
    """
    # Liên kết sampler với một mô hình ConditionalGaussianDiffusion đã định nghĩa quá trình forward (tạo noise)
    def __init__(self, diffusion: ConditionalGaussianDiffusion, config: SamplerConfig):
        self.d = diffusion
        self.config = config
        self.device = config.device or diffusion.device
        self.null_context_cache = {} # Cache cho các batch size khác nhau

    def make_time_schedule(self, steps: int) -> List[int]:
        # linear sampling of timesteps from T-1 down to 0
        T = self.d.timesteps
        times = list(range(0, T, max(1, T // steps)))
        if times[-1] != T-1:
            times.append(T-1)
        times = sorted(list(set(times))) 
        return times[::-1]  # descending

    def _get_null_context(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Helper để cache null_context cho sampling."""
        if batch_size in self.null_context_cache:
            return self.null_context_cache[batch_size]
        
        print(f"Caching null context for sampling (batch size {batch_size})...")
        with torch.no_grad():
            _, null_ctx = self.d.text_embedder.encode([""] * batch_size)
        null_ctx = null_ctx.to(device)
        self.null_context_cache[batch_size] = null_ctx
        return null_ctx
    
    @torch.no_grad()
    def sample(self, batch_size: int, model: nn.Module, context: Optional[torch.Tensor] = None, config: Optional[SamplerConfig] = None) -> torch.Tensor:
        """
        DDIM-style sampler with optional classifier-free guidance (CFG).

        - batch_size: number of samples to produce
        - model: callable model(x, t, context=None) -> eps prediction (B,C,H,W)
        - context: sequence embeddings tensor (B, L, D) or None
        - Returns: tensor of images in model's pixel range (usually [-1,1])
        """
        """
        DDIM Sampler đã tối ưu (CFG-Sample & AMP).
        Trả về latent z.
        """
        cfg = config or self.config
        device = cfg.device or self.device
        times = self.make_time_schedule(cfg.steps)
        B = batch_size
        
        # Bắt đầu từ nhiễu z_T trong latent space
        shape = (B, self.d.channels, self.d.image_size, self.d.image_size)
        x = torch.randn(shape, device=device) # x bây giờ là z

        use_guidance = (context is not None) and (cfg.guidance_scale > 1.0)
        guidance_scale = cfg.guidance_scale if use_guidance else 1.0

        if context is not None:
            context = context.to(device)
        
        if use_guidance:
            null_context = self._get_null_context(B, device)

        # Bọc toàn bộ vòng lặp bằng autocast (AMP)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device, enabled=True): #autocast(enabled=True):
            for idx in range(len(times) - 1):
                t_i = times[idx]
                t_next_i = times[idx + 1]
                t = torch.full((B,), t_i, dtype=torch.long, device=device)
                t_next = torch.full((B,), t_next_i, dtype=torch.long, device=device)

                # ===== TỐI ƯU HÓA CFG-SAMPLE (Concatenation Trick) =====
                if use_guidance:
                    # 1. Ghép batch input [B, ...] -> [2B, ...]
                    x_in = torch.cat([x, x], dim=0)
                    t_in = torch.cat([t, t], dim=0)
                    
                    # 2. Ghép context [uncond | cond]
                    context_in = torch.cat([null_context, context], dim=0)

                    # 3. Chạy model MỘT LẦN
                    eps_all = model(x_in, t_in, context_in) # (2B, C, H, W)

                    # 4. Tách kết quả
                    eps_uncond, eps_cond = torch.chunk(eps_all, 2, dim=0)

                    # 5. Combine (CFG)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    # Chạy 1 lần (không CFG)
                    eps = model(x, t, context)
                
                # ===== (Kết thúc khối tối ưu) =====

                # ===== DDIM update (giữ nguyên) =====
                alpha_t = self.d.alphas_cumprod[t].view(-1, 1, 1, 1)
                alpha_next = self.d.alphas_cumprod[t_next].view(-1, 1, 1, 1)
                sqrt_alpha_t = alpha_t.sqrt()
                sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()

                x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0) # Clamp latent

                eta = float(cfg.eta)
                sigma = eta * torch.sqrt(
                    (1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t)
                )
                coef = torch.sqrt(1.0 - alpha_next)
                dir_part = coef * eps
                
                if eta > 0:
                    noise = torch.randn_like(x) * sigma
                else:
                    noise = torch.zeros_like(x)

                x = torch.sqrt(alpha_next) * x0_pred + dir_part + noise # (Phải là + noise)
        
        return x # Trả về latent z_0

# code bên dưới hầu như không cần thiết 
# -------------------- Helper: Batch-guided model call --------------------

def model_call_guided(model: nn.Module, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor], guidance_scale: float) -> torch.Tensor:
    """A helper that implements the concatenation trick for classifier-free guidance.
    It expects `context` to be either None or a tensor (B, L, D). If context is provided,
    it concatenates unconditioned & conditioned batches and runs a single forward pass.
    Returns eps of shape (B, C, H, W) after applying guidance.
    """
    if context is None or guidance_scale == 1.0:
        return model(x, t, context)
    B = x.shape[0]
    # duplicate batch: [uncond | cond]
    x_in = torch.cat([x, x], dim=0)
    # time tensor duplicated
    t_in = torch.cat([t, t], dim=0)
    # build contexts: first half None -> model must accept None per example; we instead pass a zero context and rely on model
    # to detect 'None' via an extra flag. Many implementations accept context=None and handle it internally.
    # For generality, require model to accept 'context' where unconditioned is None.

    # prepare context batch where first half are None (we'll pass None instead by calling twice), but to be efficient
    # some models accept a context tensor and a mask; here we assume model supports None/actual mixing via separate calls.
    eps_uncond = model(x, t, None)
    eps_cond = model(x, t, context)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


# -------------------- PLMS/PNDM skeleton (multi-step predictor-corrector) --------------------

class PNDMSampler:
    """Skeleton for Pseudo Numerical Methods for Diffusion Models (PNDM) / PLMS.
    Implementation note: full PLMS requires maintaining several previous eps predictions and
    using linear multistep integration. Here we provide a simplified multistep integrator.
    """
    def __init__(self, diffusion: ConditionalGaussianDiffusion, config: SamplerConfig):
        self.d = diffusion
        self.config = config
        self.device = config.device or diffusion.device

    @torch.no_grad()
    def sample(self, batch_size: int, model: nn.Module, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # This is a placeholder showing where a full PNDM implementation would go.
        # For now, we fall back to DDIM behavior.
        ddim = DDIMSampler(self.d, self.config)
        return ddim.sample(batch_size, model, context, self.config)


# -------------------- High-level sampler factory --------------------

def make_sampler(diffusion: ConditionalGaussianDiffusion, config: SamplerConfig):
    if config.sampler == 'ddim':
        return DDIMSampler(diffusion, config)
    elif config.sampler == 'pndm':
        return PNDMSampler(diffusion, config)
    else:
        raise ValueError('Unknown sampler')


# -------------------- Example usage function --------------------

def sample_images(model: nn.Module, diffusion: ConditionalGaussianDiffusion, text_encoder, prompts: List[str], batch_size: int = 4,
                  sampler_config: Optional[SamplerConfig] = None, ema: Optional[EMA] = None) -> torch.Tensor:
    """Convenience function: apply EMA weights (if provided), run sampler, restore weights.
    Returns tensor of images in [-1,1].
    text_encoder: wrapper that turns list[str] -> seq_emb tensor (B, L, D)
    """
    cfg = sampler_config or SamplerConfig()
    device = cfg.device or diffusion.device
    model.to(device)

    if ema is not None:
        ema.apply_shadow()

    # encode prompts
    if prompts is None or len(prompts) == 0:
        context = None
    else:
        _, context = text_encoder.encode(prompts)

    sampler = make_sampler(diffusion, cfg)
    with torch.no_grad():
        # if guidance required, sampler may expect context; for concat-trick use model_call_guided
        samples = sampler.sample(batch_size=batch_size, model=model, context=context, config=cfg)

    if ema is not None:
        ema.restore()

    return samples
