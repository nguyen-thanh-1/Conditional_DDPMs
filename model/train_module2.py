# train_module2.py
"""
Kịch bản (script) chính để huấn luyện Module 2 (ControlNet).
*** ĐÃ SỬA ĐỔI ĐỂ SỬ DỤNG TẬP VALIDATION CHO VIỆC ĐÁNH GIÁ ***
"""
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from typing import List, Dict, Any

# --- Import các thư viện Diffusers ---
try:
    from diffusers import (
        AutoencoderKL, 
        UNet2DConditionModel,
        ControlNetModel,
        DPMSolverMultistepScheduler # Sampler nhanh cho evaluation
    )
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("Lỗi: Vui lòng cài đặt thư viện. Chạy: pip install diffusers transformers accelerate")
    exit()

# --- Import các file Module 2 ---
from data_parser_module2 import get_image_info_from_json, COCOSpatialDataset
from Conditioning import (
    CLIPTextEmbedder, 
    ControlNetSpatialEmbedder, 
    ConditionalGaussianDiffusion
)
from Evaluation_Logging import Metrics, TrainingLogger, Evaluator
from Trainer import Trainer # Lớp Trainer (đã sửa)

# -----------------------------------------------------------------
# LỚP CẤU HÌNH (SỬA CÁC THAM SỐ CỦA BẠN Ở ĐÂY)
# -----------------------------------------------------------------
class TrainingConfig:
    def __init__(self):
        # === (BẮT BUỘC) ĐƯỜNG DẪN DỮ LIỆU ===
        
        # 1. Sửa đường dẫn đến thư mục gốc của dataset
        self.root_dir = r"C:\Users\PC\OneDrive\Desktop\scientific research\model\training_dataset_filtered" # <-- SỬA Ở ĐÂY
        
        # --- (SỬA ĐỔI) Cấu hình tập TRAIN ---
        self.train_image_dir_name = "train_dataset"
        self.train_annotation_dir_name = "annotations"
        self.train_captions_file = "train_captions_train2017.json"
        self.train_instances_file = "train_instances_train2017.json"

        # --- (MỚI) Cấu hình tập VAL (Dùng để đánh giá) ---
        self.val_image_dir_name = "val_dataset" # <-- SỬA Ở ĐÂY (ví dụ: "val2017_png" hoặc "valInVal")
        self.val_annotation_dir_name = "annotations" # (Thường dùng chung)
        self.val_captions_file = "val_captions_train2017.json" # <-- SỬA Ở ĐÂY
        self.val_instances_file = "val_instances_train2017.json" # <-- SỬA Ở ĐÂY
        

        # === (TÙY CHỌN) ĐƯỜNG DẪN LƯU TRỮ ===
        self.checkpoint_dir = "./checkpoints_module2"
        self.log_dir = "./logs_module2"
        self.checkpoint_name = "controlnet_sketch"
        self.resume_checkpoint = None
        
        # === CẤU HÌNH MÔ HÌNH (Stable Diffusion 1.5) ===
        self.model_name = "runwayml/stable-diffusion-v1-5"
        self.vae_subfolder = "vae"
        self.unet_subfolder = "unet"
        self.text_encoder_subfolder = "text_encoder"
        
        # === (SỬA ĐỔI) CẤU HÌNH HUẤN LUYỆN (Theo gợi ý) ===
        self.image_size = 256
        self.epochs = 50 # (Gợi ý: Tăng epochs)
        self.batch_size = 2    # (Gợi ý: Giảm nếu OOM)
        self.lr = 1e-4
        self.use_amp = True
        self.grad_accum_steps = 4 # (Gợi ý: Tăng nếu giảm batch_size)
        
        # === CẤU HÌNH LOGGING/EVALUATION ===
        self.eval_batch_size = 4
        self.save_every_n_steps = 1000
        self.eval_every_n_steps = 1000
        
        # (Prompt để sinh ảnh thử nghiệm - ĐÃ SỬA ĐỔI)
        self.eval_prompts = [
            "an elephant on the left and a bird on the right", # <-- Sửa
            "a cow in a field",                              # <-- Sửa
            "a bird flying over an elephant",                # <-- Sửa
            "two giraffes next to a tree"                    # (Giữ nguyên)
        ]
        # (Box/Label tương ứng - ĐÃ SỬA ĐỔI)
        self.eval_boxes = [
            torch.tensor([[0.2, 0.1, 0.8, 0.4], [0.3, 0.6, 0.5, 0.9]]), # elephant, bird
            torch.tensor([[0.3, 0.3, 0.7, 0.7]]), # cow
            torch.tensor([[0.2, 0.2, 0.5, 0.5], [0.4, 0.1, 0.9, 0.9]]), # bird, elephant
            torch.tensor([[0.1, 0.1, 0.8, 0.4], [0.1, 0.6, 0.8, 0.9]]) # giraffe, giraffe
        ]
        self.eval_labels = [
            ["elephant", "bird"],   # <-- Sửa
            ["cow"],                # <-- Sửa
            ["bird", "elephant"],   # <-- Sửa
            ["giraffe", "giraffe"]  # (Giữ nguyên)
        ]
        
# -----------------------------------------------------------------
# HÀM CUSTOM COLLATE (Không cần sửa)
# -----------------------------------------------------------------

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Xử lý batch có số lượng box/prompt không đồng đều.
    """
    images = torch.stack([item['images'] for item in batch])
    prompts = [item['prompts'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        "images": images,
        "prompts": prompts,
        "boxes": boxes,
        "labels": labels
    }

# -----------------------------------------------------------------
# HÀM HUẤN LUYỆN CHÍNH (ĐÃ SỬA ĐỔI)
# -----------------------------------------------------------------
def main_train(config: TrainingConfig):
    
    # --- 1. Thiết lập cơ bản ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- [Module 2] Đang sử dụng thiết bị: {device} ---")
    
    latent_size_H = config.image_size // 8
    latent_size_W = config.image_size // 8
    print(f"Kích thước ảnh: {config.image_size}, Kích thước Latent: {(latent_size_H, latent_size_W)}")

    # --- (SỬA ĐỔI) Đường dẫn file (Tách biệt Train và Val) ---
    # Đường dẫn Train
    train_image_dir = os.path.join(config.root_dir, config.train_image_dir_name)
    train_captions_file = os.path.join(config.root_dir, config.train_annotation_dir_name, config.train_captions_file)
    train_instances_file = os.path.join(config.root_dir, config.train_annotation_dir_name, config.train_instances_file)

    # Đường dẫn Val
    val_image_dir = os.path.join(config.root_dir, config.val_image_dir_name)
    val_captions_file = os.path.join(config.root_dir, config.val_annotation_dir_name, config.val_captions_file)
    val_instances_file = os.path.join(config.root_dir, config.val_annotation_dir_name, config.val_instances_file)


    # --- 2. Tải và chuẩn bị Dữ liệu (Tập TRAIN) ---
    print("--- [Module 2] Đang tải dữ liệu TẬP HUẤN LUYỆN (TRAIN)... ---")
    try:
        train_img_paths, _, train_img_captions, train_img_boxes = get_image_info_from_json(train_captions_file, train_instances_file)
    except FileNotFoundError as e:
        print(f"Lỗi nghiêm trọng (TRAIN): {e}")
        print(f"Vui lòng kiểm tra lại 'root_dir' và cấu hình tập TRAIN.")
        return

    # Định nghĩa Image Transforms
    transform = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(), # Chuyển ảnh sang [0, 1]
    ])

    dataset = COCOSpatialDataset(
        image_dir=train_image_dir,
        image_id_to_path=train_img_paths,
        image_id_to_captions=train_img_captions,
        image_id_to_boxes=train_img_boxes,
        transform=transform,
        max_captions_per_image=1
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn, # <-- Rất quan trọng
        pin_memory=True
    )
    print(f"--- [Module 2] Đã tạo Dataloader TẬP TRAIN với {len(dataloader)} batches. ---")

    # --- 3. (SỬA ĐỔI) Chuẩn bị Batch Đánh giá (Từ tập VALIDATION) ---
    print("--- [Module 2] Đang tải dữ liệu TẬP ĐÁNH GIÁ (VAL)... ---")
    try:
        val_img_paths, _, val_img_captions, val_img_boxes = get_image_info_from_json(val_captions_file, val_instances_file)
        
        eval_dataset = COCOSpatialDataset(
            image_dir=val_image_dir, # <-- Dùng thư mục ảnh VAL
            image_id_to_path=val_img_paths,
            image_id_to_captions=val_img_captions,
            image_id_to_boxes=val_img_boxes,
            transform=transform,
            max_captions_per_image=1
        )
        print("--- [Module 2] Đã tải tập VAL thành công. Đánh giá sẽ dùng dữ liệu unseen.")

    except FileNotFoundError as e:
        print(f"Lỗi nghiêm trọng (VAL): {e}")
        print(f"Không tìm thấy file validation. Đánh giá FID/CLIP sẽ không chính xác.")
        # (Nếu không tìm thấy, quay lại dùng tập train để tránh crash)
        print("Cảnh báo: Sử dụng tạm tập TRAIN để đánh giá. Chỉ số sẽ bị sai lệch.")
        eval_dataset = dataset # <-- DÙNG LẠI TẬP TRAIN
        

    # Lấy 1 batch thật từ dataloader VAL để làm ảnh "real"
    real_eval_batch = next(iter(DataLoader(
        eval_dataset, # <-- Dùng dataset VAL (hoặc fallback là train)
        batch_size=config.eval_batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )))
    real_images_for_eval = real_eval_batch["images"].to(device) # [0, 1]

    metrics = Metrics(device)
    evaluator = Evaluator(
        metrics=metrics,
        prompts=config.eval_prompts[:config.eval_batch_size], # Dùng prompt tự định nghĩa
        boxes=config.eval_boxes[:config.eval_batch_size],     # Dùng box tự định nghĩa
        labels=config.eval_labels[:config.eval_batch_size],   # Dùng label tự định nghĩa
        real_images=real_images_for_eval # <-- Ảnh thật LẤY TỪ TẬP VAL
    )
    logger = TrainingLogger(config.log_dir)

    # --- 4. Khởi tạo Models (Giữ nguyên) ---
    print("--- [Module 2] Đang tải các mô hình (VAE, UNet, TextEncoder)... ---")
    dtype = torch.float16 if config.use_amp else torch.float32

    vae = AutoencoderKL.from_pretrained(config.model_name, subfolder=config.vae_subfolder, torch_dtype=dtype).to(device)
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder=config.unet_subfolder, torch_dtype=dtype).to(device)
    unet.requires_grad_(False)
    text_embedder = CLIPTextEmbedder(model_name=config.model_name, subfolder=config.text_encoder_subfolder, torch_dtype=dtype).to(device)
    text_embedder.requires_grad_(False)
    
    spatial_embedder = ControlNetSpatialEmbedder(
        output_size=(latent_size_H, latent_size_W),
        num_channels=5
    ).to(device)

    print("--- [Module 2] Đang khởi tạo ControlNet từ UNet... ---")
    control_net = ControlNetModel.from_unet(
        unet,
        conditioning_channels=5  # <-- THAM SỐ ĐÚNG LÀ ĐÂY
    ).to(device)

    # --- 5. Khởi tạo Scheduler (Giữ nguyên) ---
    scheduler = DPMSolverMultistepScheduler.from_config(config.model_name, subfolder="scheduler")

    # --- 6. Khởi tạo Diffusion Process (Giữ nguyên) ---
    diffusion_process = ConditionalGaussianDiffusion(
        model=unet.float(),
        control_net=control_net.float(),
        text_embedder=text_embedder.float(),
        spatial_embedder=spatial_embedder.float(),
        vae=vae.float(),
        timesteps=1000,
        beta_schedule="cosine",
        loss_type="l1",
        latent_size=(latent_size_H, latent_size_W)
    )

    # --- 7. Khởi tạo Optimizer (Giữ nguyên) ---
    trainable_params = list(control_net.parameters()) + list(spatial_embedder.parameters())
    optimizer = AdamW(trainable_params, lr=config.lr)
    
    print(f"--- [Module 2] Số lượng tham số huấn luyện (ControlNet + SpatialEmbedder): {sum(p.numel() for p in trainable_params if p.requires_grad):,} ---")

    # --- 8. Khởi tạo Trainer (Giữ nguyên) ---
    trainer = Trainer(
        diffusion=diffusion_process,
        dataloader=dataloader, # <-- Dataloader tập TRAIN
        evaluator=evaluator,   # <-- Evaluator dùng dữ liệu VAL
        logger=logger,
        optimizer=optimizer,
        scheduler=scheduler,
        lr=config.lr,
        grad_accum_steps=config.grad_accum_steps,
        use_amp=config.use_amp,
        save_every_n_steps=config.save_every_n_steps,
        eval_every_n_steps=config.eval_every_n_steps,
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_name=config.checkpoint_name,
    )

    # --- 9. Tải Checkpoint (Giữ nguyên) ---
    if config.resume_checkpoint:
        trainer.load_checkpoint(config.resume_checkpoint)

    # --- 10. BẮT ĐẦU HUẤN LUYỆN (Giữ nguyên) ---
    print("==============================================")
    print("===       BẮT ĐẦU HUẤN LUYỆN MODULE 2      ===")
    print("==============================================")
    trainer.train(epochs=config.epochs)

# -----------------------------------------------------------------
# ĐIỂM BẮT ĐẦU CHẠY (Không cần sửa)
# -----------------------------------------------------------------
if __name__ == "__main__":
    config = TrainingConfig()
    
    print("--- [Module 2] Bắt đầu chạy với cấu hình sau: ---")
    print(f"  Dataset Root: {config.root_dir}")
    print(f"  Train Images: {config.train_image_dir_name} ({config.train_captions_file})")
    print(f"  Val Images:   {config.val_image_dir_name} ({config.val_captions_file}) (Dùng để đánh giá)")
    print(f"  Checkpoint Dir: {config.checkpoint_dir}")
    print(f"  Model: {config.model_name}")
    print(f"  Image Size: {config.image_size}, Batch Size: {config.batch_size}")
    print(f"  Grad Accum: {config.grad_accum_steps} (Effective Batch: {config.batch_size * config.grad_accum_steps})")
    print(f"  Epochs: {config.epochs}, LR: {config.lr}")
    print(f"  AMP (float16): {config.use_amp}")
    print("--------------------------------------------------")

    main_train(config)