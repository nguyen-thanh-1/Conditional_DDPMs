# train_module1.py
"""
Kịch bản (script) chính để huấn luyện Module 1 (Text-to-Layout).
*** ĐÃ SỬA ĐỔI ĐỂ HỖ TRỢ RESUME (HUẤN LUYỆN TIẾP) ***
"""
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np # Cần để tính trung bình

# Import từ các file module 1
from model_module1 import TextToLayoutModel
from data_parser_module1 import CocoTextLayoutDataset, get_image_size_info

# -----------------------------------------------------------------
# LỚP CẤU HÌNH (SỬA CÁC THAM SỐ Ở ĐÂY)
# -----------------------------------------------------------------
class TrainingConfig:
    def __init__(self):
        # === (BẮT BUỘC) ĐƯỜNG DẪN DỮ LIỆU ===
        self.annotation_dir = r"C:\Users\PC\OneDrive\Desktop\scientific research\model\training_dataset_filtered\annotations" # <-- SỬA Ở ĐÂY

        self.captions_path_train = os.path.join(self.annotation_dir, "train_captions_train2017.json")
        self.instances_path_train = os.path.join(self.annotation_dir, "train_instances_train2017.json")
        
        self.captions_path_val = os.path.join(self.annotation_dir, "val_captions_train2017.json")
        self.instances_path_val = os.path.join(self.annotation_dir, "val_instances_train2017.json")
        
        # 3. Thư mục lưu checkpoints
        self.save_dir = "./checkpoints_module1"
        self.checkpoint_name = "t2l_model_best.pt"

        # --- THAY ĐỔI: Thêm đường dẫn để resume ---
        # Đặt là None để train từ đầu
        # Đặt là "./checkpoints_module1/t2l_model_best.pt" để train tiếp
        self.resume_checkpoint = None # <-- SỬA Ở ĐÂY ĐỂ RESUME
        
        # === CẤU HÌNH MÔ HÌNH ===
        self.model_name = "google/flan-t5-base"

        # === CẤU HÌNH TRAINING ===
        self.epochs = 20
        self.batch_size = 4
        self.lr = 5e-5

# -----------------------------------------------------------------
# HÀM ĐÁNH GIÁ (Giữ nguyên)
# -----------------------------------------------------------------
@torch.no_grad()
def evaluate(model: TextToLayoutModel, val_dataloader: DataLoader, device: torch.device) -> float:
    """
    Tính toán loss trên tập validation.
    """
    model.eval() # Chuyển model sang chế độ đánh giá
    total_loss = 0
    
    progress_bar = tqdm(val_dataloader, desc="Đánh giá (Validation)", leave=False)
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        total_loss += loss.item()
        
    avg_loss = total_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    model.train() # Chuyển model TRỞ LẠI chế độ huấn luyện
    return avg_loss

# -----------------------------------------------------------------
# HÀM HUẤN LUYỆN (ĐÃ SỬA ĐỔI)
# -----------------------------------------------------------------
def train(config: TrainingConfig):
    """
    Hàm huấn luyện chính, nhận vào đối tượng TrainingConfig.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- [Module 1] Đang sử dụng thiết bị: {device} ---")

    # 1. Khởi tạo Mô hình và Tokenizer
    model = TextToLayoutModel(config.model_name).to(device)
    special_tokens = ["<box>", "</box>"]
    model.add_new_tokens(special_tokens)
    tokenizer = model.tokenizer

    # 3. Chuẩn bị Dataset (Giữ nguyên)
    try:
        print("--- Đang tải tập TRAIN ---")
        image_size_info_train = get_image_size_info(config.captions_path_train)
        train_dataset = CocoTextLayoutDataset(
            captions_file=config.captions_path_train,
            instances_file=config.instances_path_train,
            tokenizer=tokenizer,
            image_size_info=image_size_info_train,
            max_len_in=512,
            max_len_out=256
        )
        
        print("--- Đang tải tập VAL ---")
        image_size_info_val = get_image_size_info(config.captions_path_val)
        val_dataset = CocoTextLayoutDataset(
            captions_file=config.captions_path_val,
            instances_file=config.instances_path_val,
            tokenizer=tokenizer,
            image_size_info=image_size_info_val,
            max_len_in=512,
            max_len_out=256
        )
    except FileNotFoundError as e:
        print(f"Lỗi nghiêm trọng: {e}")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 4. Thiết lập Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)
    
    # --- THAY ĐỔI: Logic tải checkpoint (Load) ---
    best_val_loss = float('inf')
    start_epoch = 0
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, config.checkpoint_name)
    
    # Kiểm tra xem có resume không
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        print(f"--- [Module 1] Đang tải checkpoint từ: {config.resume_checkpoint} ---")
        try:
            checkpoint = torch.load(config.resume_checkpoint, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load metadata
            start_epoch = checkpoint['epoch'] + 1 # Bắt đầu từ epoch TIẾP THEO
            best_val_loss = checkpoint['best_val_loss']
            
            print(f"--- [Module 1] Tải hoàn tất. Tiếp tục từ Epoch {start_epoch}. Best Val Loss: {best_val_loss:.4f} ---")
        
        except Exception as e:
            print(f"LỖI: Không thể tải checkpoint. {e}")
            print("--- [Module 1] Bắt đầu huấn luyện từ đầu. ---")
            start_epoch = 0
            best_val_loss = float('inf')
            
    else:
        if config.resume_checkpoint:
            print(f"--- [Module 1] Cảnh báo: Không tìm thấy checkpoint '{config.resume_checkpoint}'. Bắt đầu từ đầu. ---")
        else:
            print("--- [Module 1] Không có checkpoint, bắt đầu huấn luyện từ đầu. ---")
    # --- KẾT THÚC THAY ĐỔI ---


    print("--- [Module 1] Bắt đầu Huấn luyện ---")

    # --- THAY ĐỔI: Bắt đầu từ start_epoch ---
    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs} (Train)", leave=True)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} hoàn tất. Trung bình Train Loss: {avg_train_loss:.4f}")

        # 5. Chạy đánh giá
        current_val_loss = evaluate(model, val_dataloader, device)
        
        # 6. (MỚI) Lưu checkpoint nếu tốt hơn
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            
            # --- THAY ĐỔI: Lưu đầy đủ trạng thái ---
            print(f"*** Val Loss cải thiện. Đang lưu checkpoint... (Loss: {best_val_loss:.4f}) ***")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            print(f"*** Đã lưu checkpoint tốt nhất tại: {save_path} ***")

    print("--- [Module 1] Huấn luyện hoàn tất ---")
    print(f"Model tốt nhất đã được lưu tại {save_path} với Val Loss: {best_val_loss:.4f}")

# -----------------------------------------------------------------
# ĐIỂM BẮT ĐẦU CHẠY (Không cần sửa)
# -----------------------------------------------------------------
if __name__ == "__main__":
    config = TrainingConfig()
    
    print("--- [Module 1] Bắt đầu chạy với cấu hình sau: ---")
    print(f"  Train Captions:  {config.captions_path_train}")
    print(f"  Train Instances: {config.instances_path_train}")
    print(f"  Val Captions:    {config.captions_path_val}")
    print(f"  Val Instances:   {config.instances_path_val}")
    print(f"  Save Checkpoint: {os.path.join(config.save_dir, config.checkpoint_name)}")
    print(f"  Model name:      {config.model_name}")
    print(f"  Epochs:          {config.epochs}, Batch Size: {config.batch_size}, LR: {config.lr}")
    # --- THAY ĐỔI: In ra trạng thái resume ---
    if config.resume_checkpoint:
        print(f"  Resume:          {config.resume_checkpoint}")
    else:
        print("  Resume:          Không (Huấn luyện từ đầu)")
    print("--------------------------------------------------")

    train(config)