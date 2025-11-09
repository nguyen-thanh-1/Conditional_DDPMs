
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# -----------------------------
# Config
# -----------------------------
model_id = "runwayml/stable-diffusion-v1-5"          # base model
fine_tuned_unet_dir = r"C:\Users\Admin\Desktop\scientific research\sd_text2sketch_finetuned-v2"  # path tới UNet fine-tuned của bạn
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load pipeline + UNet fine-tuned
# -----------------------------
print("🔹 Đang tải Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

print(f"🔹 Đang tải UNet fine-tuned từ: {fine_tuned_unet_dir}")
unet_finetuned = UNet2DConditionModel.from_pretrained(fine_tuned_unet_dir, torch_dtype=torch.float16).to(device)
pipe.unet = unet_finetuned

# -----------------------------
# Sinh ảnh từ prompt
# -----------------------------
prompt = "a realistic pencil sketch of a young woman smiling with long curly hair" #a realistic pencil sketch of 
num_inference_steps = 30

print(f"🖊️ Generating sketch for prompt: \"{prompt}\" ...")
image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]

# -----------------------------
# Lưu ảnh kết quả
# -----------------------------
out_path = "output_sketch_v5.png"
image.save(out_path)
print(f"✅ Ảnh đã lưu tại: {out_path}")

