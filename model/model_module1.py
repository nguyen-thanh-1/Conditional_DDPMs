# model_module1.py
"""
Định nghĩa kiến trúc mô hình Text-to-Layout (T2L).
Sử dụng mô hình T5ForConditionalGeneration từ thư viện transformers
để thực hiện tác vụ sequence-to-sequence (text-in, text-out).
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast, logging

# Tắt các cảnh báo không cần thiết của Hugging Face
logging.set_verbosity_error()

class TextToLayoutModel(nn.Module):
    """
    Một lớp vỏ (wrapper) đơn giản bao quanh mô hình T5.
    """
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Khởi tạo mô hình và tokenizer.

        Args:
            model_name (str): Tên của mô hình T5 (ví dụ: "t5-small", "t5-base", "google/flan-t5-base")
        """
        super().__init__()
        print(f"--- [Module 1] Đang tải mô hình T5: {model_name} ---")
        
        # Tải tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        
        # Tải mô hình
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        print("--- [Module 1] Tải T5 hoàn tất ---")

    def add_new_tokens(self, new_tokens: list) -> int:
        """
        Thêm các token đặc biệt (ví dụ: <box>, </box>) vào tokenizer
        và thay đổi kích thước embedding của mô hình cho phù hợp.
        """
        # Thêm token vào tokenizer
        self.tokenizer.add_tokens(new_tokens)
        
        # Thay đổi kích thước ma trận embedding của mô hình
        self.t5_model.resize_token_embeddings(len(self.tokenizer))
        print(f"--- [Module 1] Đã thêm tokens mới. Kích thước Tokenizer: {len(self.tokenizer)} ---")
        return len(self.tokenizer)

    def forward(self, 
              input_ids: torch.Tensor, 
              attention_mask: torch.Tensor, 
              labels: torch.Tensor) -> torch.Tensor:
        """
        Hàm forward cho quá trình training.
        """
        # T5 tự động tính toán loss nếu 'labels' được cung cấp
        outputs = self.t5_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, 
                 text_prompt: str, 
                 max_length: int = 256, 
                 num_beams: int = 4) -> str:
        """
        Hàm sinh layout cho quá trình inference.
        """
        self.t5_model.eval()
        
        # Tiền xử lý prompt đầu vào
        input_text = f"text: {text_prompt}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512, # Giới hạn độ dài prompt
            padding="max_length",
            truncation=True
        )
        
        input_ids = inputs.input_ids.to(self.t5_model.device)
        attention_mask = inputs.attention_mask.to(self.t5_model.device)

        # Sinh chuỗi token output
        output_sequences = self.t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        # Decode chuỗi token thành văn bản
        generated_text = self.tokenizer.decode(
            output_sequences[0], 
            skip_special_tokens=True
        )
        
        return generated_text