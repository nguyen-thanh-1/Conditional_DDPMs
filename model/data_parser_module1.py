# data_parser_module1.py
"""
Định nghĩa Dataset và DataModule cho Module 1.
Cần 2 file JSON từ COCO:
1. captions_train2017.json (chứa: image_id -> caption)
2. instances_train2017.json (chứa: image_id -> bboxes, category_ids)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict

class CocoTextLayoutDataset(Dataset):
    """
    Dataset tùy chỉnh để tải các cặp (prompt, layout_string) từ COCO.
    """
    def __init__(self, 
                 captions_file: str, 
                 instances_file: str, 
                 tokenizer, 
                 image_size_info: Dict[int, Tuple[int, int]],
                 max_len_in: int = 512, 
                 max_len_out: int = 256):
        """
        Args:
            captions_file (str): Đường dẫn đến captions_train2017.json
            instances_file (str): Đường dẫn đến instances_train2017.json
            tokenizer: Tokenizer của T5 đã được load
            image_size_info: Dict mapping image_id -> (width, height)
            max_len_in (int): Độ dài tối đa của chuỗi input
            max_len_out (int): Độ dài tối đa của chuỗi output (layout)
        """
        self.captions_file = captions_file
        self.instances_file = instances_file
        self.tokenizer = tokenizer
        self.image_size_info = image_size_info
        self.max_len_in = max_len_in
        self.max_len_out = max_len_out
        
        self.data_pairs = []
        
        # Bắt đầu tải và xử lý dữ liệu
        self._load_data()

    def _load_data(self):
        """
        Hàm hoàn chỉnh để xử lý 2 file JSON và tạo các cặp dữ liệu.
        """
        print("--- [Module 1] Bắt đầu xử lý dữ liệu COCO... ---")
        
        # 1. Kiểm tra file
        if not os.path.exists(self.instances_file):
            raise FileNotFoundError(
                f"Không tìm thấy file '{self.instances_file}'. "
                "Vui lòng tải file này lên và cung cấp đúng đường dẫn."
            )
        if not os.path.exists(self.captions_file):
            raise FileNotFoundError(f"Không tìm thấy file '{self.captions_file}'.")

        # 2. Tải file instances để lấy categories và bboxes
        print(f"Đang tải {self.instances_file}...")
        with open(self.instances_file, 'r') as f:
            instances_data = json.load(f)
        
        # 3. Tạo map: category_id -> category_name
        # (ví dụ: {1: "person", 2: "bicycle", ...})
        cat_id_to_name = {cat['id']: cat['name'] for cat in instances_data.get('categories', [])}
        print(f"Tìm thấy {len(cat_id_to_name)} categories.")

        # 4. Tạo map: image_id -> list of boxes
        # (Mỗi box cần được chuẩn hóa về [0, 1])
        image_id_to_boxes = defaultdict(list)
        print("Đang xử lý annotations (bounding boxes)...")
        for ann in tqdm(instances_data.get('annotations', [])):
            image_id = ann['image_id']
            if image_id not in self.image_size_info:
                continue # Bỏ qua nếu không có thông tin kích thước ảnh
            
            img_w, img_h = self.image_size_info[image_id]
            if img_w == 0 or img_h == 0:
                continue # Bỏ qua ảnh lỗi
                
            cat_id = ann['category_id']
            label = cat_id_to_name.get(cat_id, "unknown")
            
            # COCO format [x, y, w, h] (top-left)
            x, y, w, h = ann['bbox']
            
            # Chuẩn hóa về [0, 1] và đổi sang [ymin, xmin, ymax, xmax]
            ymin = y / img_h
            xmin = x / img_w
            ymax = (y + h) / img_h
            xmax = (x + w) / img_w
            
            # Làm tròn để chuỗi ngắn gọn
            norm_box = [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]
            
            image_id_to_boxes[image_id].append((label, norm_box))

        del instances_data # Giải phóng bộ nhớ

        # 5. Tải file captions và tạo các cặp dữ liệu
        print(f"Đang tải {self.captions_file}...")
        with open(self.captions_file, 'r') as f:
            captions_data = json.load(f)

        print("Đang tạo các cặp (prompt, layout)...")
        for ann in tqdm(captions_data.get('annotations', [])):
            image_id = ann['image_id']
            caption = ann['caption'].strip().lower().replace('"', "'") # Dọn dẹp caption
            
            # Lấy các boxes cho image_id này
            boxes_for_image = image_id_to_boxes.get(image_id)
            
            if not boxes_for_image:
                continue # Bỏ qua nếu ảnh không có box nào

            # 6. Chuyển layout thành chuỗi
            # Định dạng: "layout: <box> label y1 x1 y2 x2 </box> ..."
            layout_str_parts = ["layout:"]
            for label, box in boxes_for_image:
                box_str = " ".join(map(str, box))
                layout_str_parts.append(f"<box> {label} {box_str} </box>")
            
            layout_string = " ".join(layout_str_parts)
            
            # Định dạng input cho T5
            input_string = f"text: {caption}"
            
            self.data_pairs.append((input_string, layout_string))

        print(f"--- [Module 1] Xử lý hoàn tất. Đã tạo {len(self.data_pairs)} cặp dữ liệu. ---")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.data_pairs[idx]

        # Tokenize input
        tokenized_input = self.tokenizer(
            input_text,
            max_length=self.max_len_in,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target (labels)
        tokenized_target = self.tokenizer(
            target_text,
            max_length=self.max_len_out,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Thay thế pad_token_id bằng -100 để hàm loss bỏ qua
        labels = tokenized_target.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized_input.input_ids.squeeze(0),
            "attention_mask": tokenized_input.attention_mask.squeeze(0),
            "labels": labels.squeeze(0)
        }

def get_image_size_info(captions_file: str) -> Dict[int, Tuple[int, int]]:
    """
    Hàm tiện ích để lấy (width, height) từ 'images' 
    trong file captions_train2017.json.
    """
    print("--- [Module 1] Đang tải thông tin kích thước ảnh... ---")
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Không tìm thấy file '{captions_file}'.")
        
    with open(captions_file, 'r') as f:
        data = json.load(f)
    
    image_info = {}
    for img in data.get('images', []):
        image_info[img['id']] = (img['width'], img['height'])
    
    print(f"--- [Module 1] Đã tải kích thước của {len(image_info)} ảnh. ---")
    return image_info