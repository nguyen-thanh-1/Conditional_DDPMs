# inference_pipeline.py
"""
Module 3: Pipeline Inference Hoàn chỉnh (Text-to-Image).
Kết nối Module 1 (T2L) và Module 2 (L2I)
"""
import torch
import os
import re # Thư viện Regular Expression để xử lý chuỗi layout
from PIL import Image
from typing import List, Tuple
import tqdm
# --- Import các model từ Module 1 và 2 ---
try:
    from model_module1 import TextToLayoutModel
    from Conditioning import CLIPTextEmbedder, ControlNetSpatialEmbedder
except ImportError:
    print("Lỗi: Không tìm thấy 'model_module1.py' hoặc 'Conditioning.py'.")
    print("Vui lòng đảm bảo 2 file đó ở cùng thư mục với file này.")
    exit()
    
# --- Import các thư viện Diffusers ---
try:
    from diffusers import (
        AutoencoderKL, 
        UNet2DConditionModel,
        ControlNetModel,
        DPMSolverMultistepScheduler
    )
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("Lỗi: Vui lòng cài đặt thư viện. Chạy: pip install diffusers transformers accelerate")
    exit()

# -----------------------------------------------------------------
# LỚP CẤU HÌNH (SỬA CÁC THAM SỐ CỦA BẠN Ở ĐÂY)
# -----------------------------------------------------------------
class InferenceConfig:
    def __init__(self):
        # === (BẮT BUỘC) ĐƯỜNG DẪN CHECKPOINT ===
        
        # 1. Checkpoint của Module 1 (Text-to-Layout)
        self.module1_ckpt_path = r"./checkpoints_module1/t2l_model_epoch_20.pt" # <-- SỬA Ở ĐÂY
        
        # 2. Checkpoint của Module 2 (ControlNet)
        self.module2_ckpt_path = r"./checkpoints_module2/controlnet_sketch_epoch_50.pt" # <-- SỬA Ở ĐÂY

        # === CẤU HÌNH MÔ HÌNH (Phải giống lúc train) ===
        self.base_model_name = "runwayml/stable-diffusion-v1-5"
        self.t5_model_name = "google/flan-t5-base" # Model T5 (Module 1)
        
        # === CẤU HÌNH SINH ẢNH ===
        self.image_size = 256
        
        # (Prompt bạn muốn sinh ảnh)
        self.prompt = "a cat on the left, a dog in the middle, and a tree on the right" # <-- SỬA Ở ĐÂY
        
        self.output_filename = "generated_image.png" # Tên file ảnh kết quả
        
        # Cấu hình Sampler
        self.num_inference_steps = 50
        self.cfg_scale_text = 7.5 # (Mức độ bám sát Text)
        
# -----------------------------------------------------------------
# HÀM HELPER: XỬ LÝ CHUỖI LAYOUT (Không cần sửa)
# -----------------------------------------------------------------
def parse_layout_string(layout_string: str) -> Tuple[torch.Tensor, List[str]]:
    """
    Hàm phân tích chuỗi layout do Module 1 tạo ra.
    Input: "layout: <box> cat 0.1 0.1 0.4 0.4 </box> <box> tree ... </box>"
    Output: (Tensor BBoxes, List Labels)
    """
    boxes = []
    labels = []
    
    # Sử dụng Regex để tìm tất cả các thẻ <box> ... </box>
    # (.*?) nghĩa là "khớp với mọi thứ" (non-greedy)
    matches = re.findall(r"<box>(.*?)</box>", layout_string)
    
    if not matches:
        print("Cảnh báo: Module 1 không tạo ra layout nào.")
        return torch.empty(0, 4), []

    for item in matches:
        parts = item.strip().split()
        if len(parts) == 5:
            label = parts[0]
            try:
                # Lấy 4 tọa độ
                coords = [float(p) for p in parts[1:]]
                labels.append(label)
                boxes.append(coords)
            except ValueError:
                print(f"Cảnh báo: Bỏ qua box lỗi: '{item}'")
        else:
            print(f"Cảnh báo: Bỏ qua box định dạng sai: '{item}'")

    if not boxes:
        print("Cảnh báo: Không tìm thấy box hợp lệ nào.")
        return torch.empty(0, 4), []
        
    return torch.tensor(boxes, dtype=torch.float32), labels

# -----------------------------------------------------------------
# HÀM SINH ẢNH CHÍNH (Không cần sửa)
# -----------------------------------------------------------------
@torch.no_grad()
def main_generate(config: InferenceConfig):
    
    # --- 1. Thiết lập cơ bản ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"--- [Module 3] Bắt đầu Inference trên thiết bị: {device} ({torch_dtype}) ---")
    
    latent_size_H = config.image_size // 8
    latent_size_W = config.image_size // 8

    # --- 2. Tải Module 1 (Text-to-Layout) ---
    print(f"Đang tải Module 1 (T5) từ: {config.t5_model_name}")
    module1_t2l = TextToLayoutModel(config.t5_model_name).to(device)
    module1_t2l.add_new_tokens(["<box>", "</box>"]) # Rất quan trọng
    
    if not os.path.exists(config.module1_ckpt_path):
        print(f"LỖI: Không tìm thấy checkpoint Module 1: {config.module1_ckpt_path}")
        return
        
    print(f"Đang tải trọng số Module 1 từ: {config.module1_ckpt_path}")
    module1_t2l.load_state_dict(torch.load(config.module1_ckpt_path, map_location=device))
    module1_t2l.eval()

    # --- 3. Tải Module 2 (Layout-to-Image) ---
    print(f"Đang tải Module 2 (SD 1.5) từ: {config.base_model_name}")
    # 3a. Tải các thành phần đóng băng (VAE, UNet, TextEncoder)
    vae = AutoencoderKL.from_pretrained(
        config.base_model_name, subfolder="vae", torch_dtype=torch_dtype
    ).to(device)
    
    text_embedder = CLIPTextEmbedder(
        model_name=config.base_model_name, 
        subfolder="text_encoder",
        torch_dtype=torch_dtype
    ).to(device)
    
    unet = UNet2DConditionModel.from_pretrained(
        config.base_model_name, subfolder="unet", torch_dtype=torch_dtype
    ).to(device)
    
    # 3b. Tải ControlNet (đã huấn luyện)
    print("Đang khởi tạo ControlNet...")
    control_net = ControlNetModel.from_unet(
        unet, 
        conditioning_channels=5, # <-- THAM SỐ ĐÚNG LÀ ĐÂY
        torch_dtype=torch_dtype
    ).to(device)
        # 3c. Tải Spatial Embedder
    spatial_embedder = ControlNetSpatialEmbedder(
        output_size=(latent_size_H, latent_size_W), num_channels=5
    ).to(device)

    if not os.path.exists(config.module2_ckpt_path):
        print(f"LỖI: Không tìm thấy checkpoint Module 2: {config.module2_ckpt_path}")
        return
        
    print(f"Đang tải trọng số Module 2 từ: {config.module2_ckpt_path}")
    # Tải checkpoint của ControlNet (từ file Trainer.py)
    ckpt = torch.load(config.module2_ckpt_path, map_location=device)
    control_net.load_state_dict(ckpt["control_net"]) # Chỉ tải control_net
    if "spatial_embedder" in ckpt: # <-- THÊM KHỐI LỆNH NÀY
        spatial_embedder.load_state_dict(ckpt["spatial_embedder"])
        print("Đã tải trọng số 'spatial_embedder' từ checkpoint.")
    else:
        print("Cảnh báo: Không tìm thấy 'spatial_embedder' trong checkpoint. Sử dụng trọng số khởi tạo.")

    control_net.eval()
    
    spatial_embedder.eval()
    
    # 3d. Tải Scheduler
    scheduler = DPMSolverMultistepScheduler.from_config(config.base_model_name, subfolder="scheduler")
    print("--- Tải mô hình hoàn tất ---")

    # ====================================================
    # === BƯỚC 1: CHẠY MODULE 1 (Tạo Layout)
    # ====================================================
    print(f"\n[Bước 1/3] Đang chạy Module 1 (Text-to-Layout)...")
    print(f"  Prompt đầu vào: '{config.prompt}'")
    
    layout_string = module1_t2l.generate(
        text_prompt=config.prompt,
        max_length=256
    )
    
    print(f"  Layout được tạo: '{layout_string}'")

    # ====================================================
    # === BƯỚC 2: XỬ LÝ LAYOUT (Parse)
    # ====================================================
    print(f"\n[Bước 2/3] Đang xử lý chuỗi layout...")
    
    boxes_tensor, labels_list = parse_layout_string(layout_string)
    
    if boxes_tensor.shape[0] == 0:
        print("Không có layout hợp lệ, ảnh có thể không chính xác.")
    else:
        print(f"  Tìm thấy {len(labels_list)} đối tượng:")
        for l, b in zip(labels_list, boxes_tensor):
            print(f"    - {l}: {b.tolist()}")

    # ====================================================
    # === BƯỚC 3: CHẠY MODULE 2 (Sinh ảnh)
    # ====================================================
    print(f"\n[Bước 3/3] Đang chạy Module 2 (Layout-to-Image)...")

    # 1. Chuẩn bị các điều kiện (Text + Spatial)
    
    # 1a. Positive (Prompt, Boxes)
    pos_text = text_embedder([config.prompt])
    pos_spatial = spatial_embedder(boxes_tensor.to(device), labels_list, device)
    
    # 1b. Negative (Null)
    neg_text = text_embedder([""]) # Null text
    null_boxes = torch.empty(0, 4).to(device)
    neg_spatial = spatial_embedder(null_boxes, [], device) # Null spatial
    
    # 1c. Gộp lại cho CFG
    # (neg, pos)
    text_input = torch.cat([neg_text, pos_text]).to(torch_dtype)
    spatial_input = torch.cat([neg_spatial, pos_spatial]).to(torch_dtype)

    # 2. Chuẩn bị Scheduler và Latent noise
    scheduler.set_timesteps(config.num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    latents = torch.randn(
        (1, 4, latent_size_H, latent_size_W), 
        device=device,
        dtype=torch_dtype
    )
    latents = latents * scheduler.init_noise_sigma

    # 3. Vòng lặp Denoising (Sampling)
    for t in tqdm(timesteps, desc="Sampling"):
        # Mở rộng latents (cho CFG)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Chạy ControlNet
        control_residuals = control_net(
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=text_input,
            controlnet_cond=spatial_input
        )
        
        # Chạy UNet
        noise_pred = unet(
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=text_input,
            down_block_residual_samples=control_residuals.down_block_res_samples,
            mid_block_residual_sample=control_residuals.mid_block_res_sample
        ).sample
        
        # Áp dụng CFG (Text)
        noise_pred_neg, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_neg + config.cfg_scale_text * (noise_pred_pos - noise_pred_neg)
        
        # Cập nhật Latent (Denoise 1 bước)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 4. Giải mã Latents -> Ảnh (Pixel)
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents.to(vae.dtype)).sample
    
    # 5. Xử lý ảnh (Chuyển sang [0, 1] -> [0, 255] -> PIL)
    image = (image / 2.0 + 0.5).clamp(0.0, 1.0) # [-1, 1] -> [0, 1]
    image = image.cpu().permute(0, 2, 3, 1).float().numpy() # (B, H, W, C)
    image_pil = Image.fromarray((image[0] * 255).astype("uint8"))
    
    # 6. Lưu ảnh
    image_pil.save(config.output_filename)
    print(f"\n--- HOÀN TẤT! ---")
    print(f"Đã lưu ảnh kết quả tại: {config.output_filename}")


# -----------------------------------------------------------------
# ĐIỂM BẮT ĐẦU CHẠY
# -----------------------------------------------------------------
if __name__ == "__main__":
    # 1. Khởi tạo cấu hình
    config = InferenceConfig()
    
    # 2. In cấu hình
    print("--- [Module 3] Bắt đầu chạy Inference với cấu hình sau: ---")
    print(f"  Prompt:         {config.prompt}")
    print(f"  Module 1 Ckpt:  {config.module1_ckpt_path}")
    print(f"  Module 2 Ckpt:  {config.module2_ckpt_path}")
    print(f"  Output File:    {config.output_filename}")
    print("--------------------------------------------------")

    # 3. Gọi hàm sinh ảnh
    main_generate(config)