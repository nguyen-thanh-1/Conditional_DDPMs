# data_parser_module2.py
"""
Định nghĩa Dataset và các hàm tiện ích cho Module 2 (Layout-to-Image).
- Tải ảnh từ thư mục.
- Tải caption từ 'captions_train2017.json'.
- Tải layout từ 'instances_train2017.json'.
- Ghép chúng lại dựa trên 'image_id'.
"""
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Optional
from collections import defaultdict

def get_image_info_from_json(captions_file: str, instances_file: str) -> Tuple[
    Dict[int, str], 
    Dict[int, Tuple[int, int]], 
    Dict[int, List[str]], 
    Dict[int, List[Tuple[str, List[float]]]]
]:
    """
    Hàm helper để xử lý 2 file JSON lớn một lần.
    
    Returns:
        image_id_to_path (Dict): image_id -> "00000012345.jpg"
        image_id_to_size (Dict): image_id -> (width, height)
        image_id_to_captions (Dict): image_id -> ["caption 1", "caption 2"]
        image_id_to_boxes (Dict): image_id -> [("cat", [y1,x1,y2,x2]), ("dog", ...)]
    """
    print("--- [Module 2] Bắt đầu xử lý file JSON (Annotations)... ---")
    
    # 1. Tải file captions
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Không tìm thấy file: {captions_file}")
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
        
    # Lấy path và size từ 'images'
    image_id_to_path = {img['id']: img['file_name'] for img in captions_data.get('images', [])}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in captions_data.get('images', [])}
    
    # Lấy captions từ 'annotations'
    image_id_to_captions = defaultdict(list)
    for ann in tqdm(captions_data.get('annotations', []), desc="Xử lý captions"):
        image_id_to_captions[ann['image_id']].append(ann['caption'])
        
    del captions_data # Giải phóng bộ nhớ
    print(f"--- [Module 2] Đã tải {len(image_id_to_path)} ảnh và {len(image_id_to_captions)} captions. ---")

    # 2. Tải file instances
    if not os.path.exists(instances_file):
        raise FileNotFoundError(f"Không tìm thấy file: {instances_file}")
    with open(instances_file, 'r', encoding='utf-8') as f:
        instances_data = json.load(f)

    # Tạo map: category_id -> category_name
    cat_id_to_name = {cat['id']: cat['name'] for cat in instances_data.get('categories', [])}
    
    # Lấy boxes từ 'annotations'
    image_id_to_boxes = defaultdict(list)
    for ann in tqdm(instances_data.get('annotations', []), desc="Xử lý instances (boxes)"):
        image_id = ann['image_id']
        if image_id not in image_id_to_size:
            continue
            
        img_w, img_h = image_id_to_size[image_id]
        if img_w == 0 or img_h == 0:
            continue
            
        cat_id = ann['category_id']
        label = cat_id_to_name.get(cat_id, "unknown")
        
        # COCO [x, y, w, h]
        x, y, w, h = ann['bbox']
        
        # Chuẩn hóa về [0, 1] và đổi sang [ymin, xmin, ymax, xmax]
        ymin = y / img_h
        xmin = x / img_w
        ymax = (y + h) / img_h
        xmax = (x + w) / img_w
        
        norm_box = [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]
        image_id_to_boxes[image_id].append((label, norm_box))
        
    del instances_data # Giải phóng bộ nhớ
    print(f"--- [Module 2] Đã xử lý {len(image_id_to_boxes)} ảnh có bounding box. ---")
    
    return image_id_to_path, image_id_to_size, image_id_to_captions, image_id_to_boxes

class COCOSpatialDataset(Dataset):
    """
    Dataset cho Module 2: Tải (Ảnh, Prompt, Layout)
    """
    def __init__(self,
                 image_dir: str, # Thư mục chứa ảnh (ví dụ: "train2017/")
                 image_id_to_path: Dict[int, str],
                 image_id_to_captions: Dict[int, List[str]],
                 image_id_to_boxes: Dict[int, List[Tuple[str, List[float]]]],
                 transform: Optional[Callable] = None,
                 max_captions_per_image: int = 1 # Chỉ dùng 1 caption cho mỗi ảnh
                ):
        self.image_dir = image_dir
        self.transform = transform
        self.max_captions = max_captions_per_image
        
        self.samples = [] # List các (image_path, caption, boxes, labels)
        
        print("--- [Module 2] Đang tạo các mẫu (samples) huấn luyện... ---")
        # Lọc các image_id CÓ CẢ caption và box
        valid_ids = set(image_id_to_captions.keys()) & set(image_id_to_boxes.keys())
        
        for image_id in tqdm(valid_ids):
           # Lấy tên file gốc từ JSON (ví dụ: "00000012345.jpg")
            original_filename = image_id_to_path[image_id]
            
            # Tách tên file và đuôi file (ví dụ: "00000012345" và ".jpg")
            base_filename = os.path.splitext(original_filename)[0]
            
            # --- THAY ĐỔI: Tạo đường dẫn file .png ---
            png_filename = base_filename + ".png"
            img_path = os.path.join(self.image_dir, png_filename)
            # --- KẾT THÚC THAY ĐỔI ---
            
            if not os.path.exists(img_path):
                # (Tùy chọn) Bật dòng print này nếu bạn muốn xem file nào bị thiếu
                # print(f"Cảnh báo: Không tìm thấy file {img_path}. Bỏ qua image_id {image_id}")
                continue # Bỏ qua nếu không tìm thấy file ảnh .png
                
            captions = image_id_to_captions[image_id]
            boxes_data = image_id_to_boxes[image_id]
            
            # Tách (label, box)
            labels = [item[0] for item in boxes_data]
            boxes = [item[1] for item in boxes_data]
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

            # Chọn 1 (hoặc N) caption
            for i in range(min(len(captions), self.max_captions)):
                caption = captions[i]
                self.samples.append((img_path, caption, boxes_tensor, labels))
                
        print(f"--- [Module 2] Dataset hoàn tất. Tổng cộng {len(self.samples)} mẫu. ---")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption, boxes, labels = self.samples[idx]
        
        # Tải ảnh
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Lỗi tải ảnh {img_path}: {e}. Trả về ảnh rỗng.")
            image = Image.new("RGB", (256, 256), (0, 0, 0)) # Trả về ảnh đen
        
        # Áp dụng transform (Resize, ToTensor, v.v.)
        if self.transform:
            image = self.transform(image) # (3, H, W) trong [0, 1]
            
        return {
            "images": image,
            "prompts": caption,
            "boxes": boxes,     # Tensor
            "labels": labels    # List[str]
        }