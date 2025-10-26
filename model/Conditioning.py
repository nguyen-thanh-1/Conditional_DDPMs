# Conditioning.py
"""
Mục đích:
1. Cung cấp Text Embedder (CLIPTextEmbedder).
2. Cung cấp ControlNetSpatialEmbedder:
   - "Vẽ" các bounding box thành một "map" (ảnh) 5-kênh.
3. Cung cấp lớp Diffusion chính (ConditionalGaussianDiffusion) (ĐÃ VIẾT LẠI):
   - Khởi tạo với UNet (model) và ControlNet (control_net).
   - Tạo null contexts cho text (embedding) và spatial (map 0).
   - Định nghĩa `loss_fn` MỚI, gọi cả 2 model và tính loss.
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import lớp GaussianDiffusion lõi (chỉ chứa q_sample)
from Diffusion_core import GaussianDiffusion

try:
    from transformers import CLIPTokenizerFast, CLIPTextModel, logging as hf_logging
    from diffusers.models import ControlNetModel, UNet2DConditionModel, AutoencoderKL
    hf_logging.set_verbosity_error()
except ImportError:
    print("Lỗi: Vui lòng cài đặt transformers/diffusers. Chạy: pip install transformers diffusers")
    exit()


# -----------------------------------------------------------------
# BƯỚC 1: TEXT EMBEDDER (CLIP)
# -----------------------------------------------------------------

class CLIPTextEmbedder(nn.Module):
    """
    Wrapper cho CLIPTextModel của Hugging Face.
    """
    def __init__(self, 
                 model_name: str = "runwayml/stable-diffusion-v1-5", 
                 subfolder: str = "text_encoder", 
                 torch_dtype=torch.float16):
        super().__init__()
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, 
            subfolder=subfolder, 
            torch_dtype=torch_dtype
        )
        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            model_name, 
            subfolder="tokenizer"
        )
        
    def forward(self, prompts: List[str]) -> torch.Tensor:
        """
        Mã hóa một list các prompt (văn bản) thành tensor embedding.
        """
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        
        # Lấy embedding (output cuối cùng của text_encoder)
        text_embeddings = self.text_encoder(text_input_ids)[0]
        return text_embeddings

# -----------------------------------------------------------------
# BƯỚC 2: SPATIAL EMBEDDER (CONTROLNET)
# -----------------------------------------------------------------

class ControlNetSpatialEmbedder(nn.Module):
    """
    "Vẽ" Bounding Boxes thành một map 5-kênh
    như được mô tả trong paper "Layout-Encoding Diffusion" (LED).
    Kênh: [object_mask, y_min, x_min, y_max, x_max]
    """
    def __init__(self, 
                 output_size: Tuple[int, int] = (64, 64), 
                 num_channels: int = 5,
                 # (Tùy chọn) Có thể thêm 1 lớp CNN nhỏ để xử lý map này
                 # nhưng hiện tại chỉ cần "vẽ" trực tiếp
                ):
        super().__init__()
        self.output_size = output_size
        self.num_channels = num_channels
        assert num_channels == 5, "Kiến trúc này yêu cầu 5 kênh"

    def forward(self, 
                boxes: torch.Tensor, 
                labels: List[str], # (Hiện tại chưa dùng labels, nhưng để sẵn)
                device: torch.device) -> torch.Tensor:
        """
        Tạo map điều kiện từ bounding boxes.
        
        Args:
            boxes (torch.Tensor): Tensor (N_boxes, 4) chứa [ymin, xmin, ymax, xmax]
                                  đã được chuẩn hóa trong [0, 1].
            labels (List[str]): List (N_boxes) các nhãn (ví dụ: "cat", "tree").
            device: Thiết bị (cuda/cpu) để tạo map.
        
        Returns:
            torch.Tensor: Map (1, 5, H, W)
        """
        H, W = self.output_size
        # Khởi tạo map 5-kênh rỗng
        spatial_map = torch.zeros(1, self.num_channels, H, W, device=device)
        
        if boxes.shape[0] == 0:
            # Nếu không có box nào (ví dụ: null context), trả về map rỗng
            return spatial_map

        # Chuyển box [0, 1] thành tọa độ pixel [0, H] và [0, W]
        boxes_pixel = boxes.clone()
        boxes_pixel[:, 0] = boxes[:, 0] * H
        boxes_pixel[:, 1] = boxes[:, 1] * W
        boxes_pixel[:, 2] = boxes[:, 2] * H
        boxes_pixel[:, 3] = boxes[:, 3] * W
        boxes_pixel = boxes_pixel.long() # Chuyển sang int

        # "Vẽ" từng box lên map
        for i in range(boxes.shape[0]):
            box_norm = boxes[i] # [y1, x1, y2, x2] normalized
            y1, x1, y2, x2 = boxes_pixel[i] # [y1, x1, y2, x2] pixel

            # Đảm bảo tọa độ nằm trong khoảng
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(H, y2), min(W, x2)
            
            if y1 >= y2 or x1 >= x2:
                continue

            # Kênh 0: Object mask (vùng box = 1.0)
            spatial_map[0, 0, y1:y2, x1:x2] = 1.0
            
            # Kênh 1-4: Tọa độ chuẩn hóa
            spatial_map[0, 1, y1:y2, x1:x2] = box_norm[0] # y_min
            spatial_map[0, 2, y1:y2, x1:x2] = box_norm[1] # x_min
            spatial_map[0, 3, y1:y2, x1:x2] = box_norm[2] # y_max
            spatial_map[0, 4, y1:y2, x1:x2] = box_norm[3] # x_max
            
        return spatial_map

# -----------------------------------------------------------------
# BƯỚC 3: LỚP DIFFUSION CHÍNH (GHI ĐÈ loss_fn)
# -----------------------------------------------------------------

class ConditionalGaussianDiffusion(GaussianDiffusion):
    """
    Lớp Diffusion có điều kiện (Text + Spatial) sử dụng ControlNet.
    Ghi đè (overrides) hàm `loss_fn`.
    """
    def __init__(
        self,
        # Models
        model: UNet2DConditionModel,       # UNet (đóng băng)
        control_net: ControlNetModel,      # ControlNet (huấn luyện)
        text_embedder: CLIPTextEmbedder,   # Text Embedder (đóng băng)
        spatial_embedder: ControlNetSpatialEmbedder, # Spatial Embedder (đóng băng)
        vae: AutoencoderKL,                # VAE (đóng băng)
        
        # Configs
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        loss_type: str = "l1",
        
        # (MỚI) Kích thước map latent (1/8 kích thước ảnh)
        latent_size: Tuple[int, int] = (32, 32), 
    ):
        super().__init__(timesteps, beta_schedule)
        
        self.model = model
        self.control_net = control_net
        self.text_embedder = text_embedder
        self.spatial_embedder = spatial_embedder
        self.vae = vae
        self.latent_size = latent_size
        
        if loss_type == "l1":
            self.loss_func = F.l1_loss
        elif loss_type == "l2":
            self.loss_func = F.mse_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        self.device = next(model.parameters()).device
        
        # --- (MỚI) Tạo "Null" Contexts cho Classifier-Free Guidance ---
        # 1. Null Text Context
        self.register_buffer(
            "null_text_context", 
            self._encode_text_prompt("") # Mã hóa một chuỗi rỗng
        )
        
        # 2. Null Spatial Context (map 5-kênh toàn số 0)
        self.register_buffer(
            "null_spatial_context_map", 
            torch.zeros(1, 5, self.latent_size[0], self.latent_size[1], device=self.device)
        )

    @torch.no_grad()
    def _encode_text_prompt(self, prompt: str) -> torch.Tensor:
        """Hàm helper để mã hóa một prompt."""
        return self.text_embedder([prompt])

    @torch.no_grad()
    def _encode_image_latents(self, images: torch.Tensor) -> torch.Tensor:
        """
        Mã hóa ảnh (pixel space) sang latent space (sử dụng VAE).
        Input: images (B, 3, H, W) trong [-1, 1]
        Output: latents (B, 4, H/8, W/8)
        """
        # Chuyển sang float16 để tiết kiệm VRAM (nếu VAE hỗ trợ)
        if hasattr(self.vae, 'dtype'):
            images = images.to(self.vae.dtype)
            
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents.float() # Trả về float32 để train

    # === HÀM LOSS_FN CHÍNH (ĐÃ GHI ĐÈ) ===
    
    def loss_fn(
        self, 
        x_start: torch.Tensor,                # Ảnh thật (pixel, [-1, 1])
        prompts: List[str],                   # List các prompt
        boxes: List[torch.Tensor],            # List các tensor box
        labels: List[List[str]],              # List các list nhãn
        text_drop_prob: float = 0.1,          # Tỷ lệ dropout Text
        spatial_drop_prob: float = 0.1,       # Tỷ lệ dropout Spatial (Layout)
    ) -> torch.Tensor:
        """
        Hàm loss cho ControlNet.
        UNet (self.model) ĐÓNG BĂNG.
        ControlNet (self.control_net) ĐƯỢC HUẤN LUYỆN.
        """
        B = x_start.shape[0]
        device = x_start.device
        
        # 1. Sample t
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        
        # 2. Mã hóa ảnh -> latent (CHỈ 1 LẦN, dùng no_grad)
        with torch.no_grad():
            latents_start = self._encode_image_latents(x_start)
        
        # 3. Sample noise & Áp dụng q_sample
        noise = torch.randn_like(latents_start)
        x_t = self.q_sample(x_start=latents_start, t=t, noise=noise) # x_t là (B, 4, H/8, W/8)
        
        # --- (MỚI) Xử lý Điều kiện (Text & Spatial) ---
        
        text_contexts_list = []
        spatial_contexts_list = []

        # (Lặp qua batch, vì mỗi sample có số lượng box khác nhau)
        for i in range(B):
            # 3a. Lấy điều kiện Text
            text_emb = self._encode_text_prompt(prompts[i]) # (1, 77, 768)
            text_contexts_list.append(text_emb)
            
            # 3b. Lấy điều kiện Spatial
            spatial_map = self.spatial_embedder(
                boxes=boxes[i].to(device), 
                labels=labels[i], 
                device=device
            ) # (1, 5, H/8, W/8)
            spatial_contexts_list.append(spatial_map)

        # Gộp batch
        text_context = torch.cat(text_contexts_list, dim=0)    # (B, 77, 768)
        spatial_context_map = torch.cat(spatial_contexts_list, dim=0) # (B, 5, H/8, W/8)

        # --- (MỚI) Áp dụng Classifier-Free Guidance DROPOUT ---
        
        # 4. Mask Text
        text_keep_mask = (torch.rand(B, device=self.device) > text_drop_prob).float().view(B, 1, 1)
        null_text_batch = self.null_text_context.repeat(B, 1, 1)
        text_context = text_context * text_keep_mask + null_text_batch * (1.0 - text_keep_mask)

        # 4. Mask Spatial
        spatial_keep_mask = (torch.rand(B, device=self.device) > spatial_drop_prob).float().view(B, 1, 1, 1)
        null_spatial_map_batch = self.null_spatial_context_map.repeat(B, 1, 1, 1)
        spatial_context_map = spatial_context_map * spatial_keep_mask + null_spatial_map_batch * (1.0 - spatial_keep_mask)
        
        # --- Chạy Model (Kiến trúc ControlNet) ---
        print(f"[DEBUG] x_t shape: {x_t.shape}")
        print(f"[DEBUG] spatial_context_map shape (before resize): {spatial_context_map.shape}")

        if spatial_context_map.shape[2:] != x_t.shape[2:]:
            spatial_context_map = torch.nn.functional.interpolate(
                spatial_context_map, size=x_t.shape[2:], mode="bilinear", align_corners=False
            )

        print(f"[DEBUG] spatial_context_map shape (after resize): {spatial_context_map.shape}")

        # 5. Chạy ControlNet (CẦN gradient)
        # ControlNet nhận x_t, t, text và map
        control_residuals = self.control_net(
            sample=x_t,
            timestep=t,
            encoder_hidden_states=text_context,
            controlnet_cond=spatial_context_map
        ) # Trả về 1 tuple (down_samples, mid_sample)

        # 6. Chạy UNet (KHÔNG cần gradient, đã đóng băng)
        with torch.no_grad():
            eps_pred = self.model(
                sample=x_t, 
                timestep=t, 
                encoder_hidden_states=text_context,
                down_block_residual_samples=control_residuals.down_block_res_samples,
                mid_block_residual_sample=control_residuals.mid_block_res_sample
            ).sample
        
        # 7. Tính loss (so sánh noise dự đoán với noise thật)
        loss = self.loss_func(eps_pred, noise)
        return loss