"""
Purpose:
- (Tối ưu hóa) Cung cấp logic training cốt lõi cho *Conditional* DDPM.
- (Tối ưu hóa) Cung cấp Text Embedder (CLIP).
- (Tối ưu hóa) Loại bỏ toàn bộ kiến trúc UNet tùy chỉnh (CrossAttention, 
  ResidualBlockWithCA, ConditionalUNet) để thay bằng thư viện.
- (Tối ưu hóa) Tinh gọn ConditionalGaussianDiffusion, chỉ giữ lại logic loss_fn
  đã tối ưu (CFG-Train) và loại bỏ logic sampling.
"""

from typing import List, Tuple, Optional
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import lớp GaussianDiffusion đã được tinh gọn
from Diffusion_core import GaussianDiffusion

# HuggingFace CLIP text encoder
try:
    from transformers import CLIPTokenizerFast, CLIPTextModel
except ImportError:
    print("Lỗi: Vui lòng cài đặt transformers. Chạy: pip install transformers")


# -------------------- Toàn bộ CrossAttention, Blocks, UNet đã BỊ XÓA --------------------
# (Sẽ được thay thế bằng diffusers.models.UNet2DConditionModel)


# -------------------- Text Encoder Wrapper (Giữ nguyên) --------------------
# (Đây là thành phần cốt lõi để lấy text embedding)

class CLIPTextEmbedder(nn.Module):
    """
    Wrapper tải CLIP tokenizer và text model từ HuggingFace.
    Đóng băng model và dùng để encode text.
    """
    def __init__(self, 
                 pretrained: str = "openai/clip-vit-large-patch14", 
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        print(f"Loading CLIPTextModel: {pretrained}...")
        self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained)
        self.model = CLIPTextModel.from_pretrained(pretrained).to(self.device)
        self.text_emb_dim = self.model.config.hidden_size  # = 768 for CLIP-L/14

        # Đóng băng text encoder
        self.model.eval() 
        for p in self.model.parameters():
            p.requires_grad = False
        print("CLIPTextModel loaded and frozen.")

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode một batch text.
        Returns:
            - input_ids: (B, L)
            - seq_emb: (B, L, D) (last_hidden_state)
        """
        toks = self.tokenizer(
            texts, 
            padding="max_length", # Pad đến câu dài nhất trong batch
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        input_ids = toks["input_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)

        out = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=False
        )
        seq_emb = out.last_hidden_state
        return input_ids, seq_emb


# -------------------- Conditional Diffusion (TỐI ƯU HÓA) --------------------
# (Kế thừa từ GaussianDiffusion đã được tinh gọn)

class ConditionalGaussianDiffusion(GaussianDiffusion):
    """
    Lớp Conditional Diffusion đã được tối ưu:
    - Kế thừa GaussianDiffusion (đã tinh gọn).
    - Chỉ ghi đè (override) hàm 'loss_fn' để thêm logic
      Classifier-Free Guidance (CFG) Training đã được tối ưu.
    - Toàn bộ logic sampling đã bị XÓA BỎ.
    """
    def __init__(self, model: nn.Module, text_embedder: CLIPTextEmbedder, **kwargs):
        super().__init__(model=model, **kwargs)
        self.text_embedder = text_embedder
        
        # --- TỐI ƯU CFG-TRAIN ---
        # Tạo và cache null_context (unconditional embedding) một lần
        print("Caching null context for CFG-Training...")
        with torch.no_grad():
             # (1, L, D)
            _, self.null_context = self.text_embedder.encode([""])
        self.null_context = self.null_context.to(self.device)
        print(f"Null context cached, shape: {self.null_context.shape}")
        # --- (Kết thúc) ---

    def loss_fn(self, x_start: torch.Tensor, captions: list, cond_drop_prob: float = 0.1) -> torch.Tensor:
        """
        Compute training loss L_simple với tối ưu hóa CFG-Train.
        - x_start: Batch ảnh/latent sạch
        - captions: list các string captions
        - cond_drop_prob: Tỷ lệ drop caption (để học unconditional)
        """
        B = x_start.shape[0]
        
        # 1. Chuẩn bị x_t (giống unconditional)
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise) # (B, C, H, W)

        # 2. Encode captions (không cần gradient)
        with torch.no_grad():
            _, seq_emb = self.text_embedder.encode(captions) # (B, L, D)

        # --- TỐI ƯU CFG-TRAIN (Masking) ---
        
        # 3. Tạo mask (B,) -> (B, 1, 1)
        #    1.0 = giữ context (conditioned)
        #    0.0 = drop (dùng null_context, unconditional)
        keep_mask = (torch.rand(B, device=self.device) > cond_drop_prob).float()
        mask = keep_mask.view(B, 1, 1) # (B, 1, 1)

        # 4. Lặp null_context cho vừa batch
        null_ctx_batch = self.null_context.repeat(B, 1, 1) # (B, L, D)

        # 5. Trộn context bằng phép toán vector hóa (nhanh)
        #    Nếu mask=1, giữ seq_emb. Nếu mask=0, dùng null_ctx_batch.
        context = seq_emb * mask + null_ctx_batch * (1.0 - mask)
        
        # 6. Chạy model MỘT LẦN với batch context đã trộn
        #    Model (UNet2DConditionModel) cần nhận (x, t, encoder_hidden_states)
        eps_pred = self.model(
            sample=x_t, 
            timestep=t, 
            encoder_hidden_states=context
        ).sample # Lấy .sample vì output của diffusers UNet là một dataclass
        
        # --- (Kết thúc) ---

        # 7. Tính loss
        return F.mse_loss(eps_pred, noise)

    # --- TOÀN BỘ CÁC HÀM SAMPLING ĐÃ BỊ XÓA ---