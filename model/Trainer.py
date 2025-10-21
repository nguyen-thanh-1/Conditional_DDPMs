import os
import copy
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from PIL import Image # <-- Thêm
from torch.cuda.amp import GradScaler, autocast # <-- Thêm

from Diffusion_core import UNet, GaussianDiffusion
from Conditioning import ConditionalUNet, CLIPTextEmbedder, ConditionalGaussianDiffusion
from Guidance_Sampling import DDIMSampler # , SamplerConfig (Nếu dùng)
from Latent_Diffusion import SimpleVAE # Chỉ import VAE
from Evaluation_Logging import TrainingLogger, Evaluator, Metrics

# -------------------
# Dataset
# -------------------
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, captions, transform=None, tokenizer=None):
        self.image_paths = image_paths # <-- Sửa: đây là list các đường dẫn
        self.captions = captions
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, cap = self.image_paths[idx], self.captions[idx]
        
        # Tải "on-the-fly"
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Using a dummy image. Error: {e}")
            img = Image.new('RGB', (32, 32), (0, 0, 0)) # Ảnh giả nếu lỗi

        if self.transform:
            img = self.transform(img) # Transform
            
        # (tokenizer thường được xử lý trong diffusion.loss_fn)
        # if self.tokenizer:
        #     cap = self.tokenizer(cap)
            
        return img, cap


# -------------------
# Trainer
# -------------------
class Trainer:
    def __init__(self, model, diffusion, vae, optimizer, sampler, dataloader, evaluator, logger,
                 ema_decay=0.999, device="cpu"):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.sampler = sampler
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.logger = logger
        self.device = device

        # --- VAE Setup ---
        self.vae = vae.to(device)
        self.vae.eval() # Đóng băng VAE
        for p in self.vae.parameters():
            p.requires_grad = False
        print("VAE is frozen and in eval mode.")

        # --- EMA ---
        self.ema_model = copy.deepcopy(model).to(device)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_decay = ema_decay
        
        # --- AMP (Tối ưu) ---
        self.scaler = torch.amp.GradScaler(device=self.device) #GradScaler()
        print(f"Trainer initialized with device: {device} and AMP.")

        self.global_step = 0

    def update_ema(self):
        with torch.no_grad():
            for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
                p_ema.data = self.ema_decay * p_ema.data + (1. - self.ema_decay) * p.data

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for i, batch in enumerate(self.dataloader):
            imgs, caps = batch
            imgs = imgs.to(self.device)

            if isinstance(caps, torch.Tensor):
                caps_device = caps.to(self.device)
            else:
                caps_device = caps

            self.optimizer.zero_grad()

            # --- TỐI ƯU HÓA PIPELINE ---
            
            # 1. Bọc forward pass bằng autocast (AMP)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type=device, enabled=True): #autocast(enabled=True):
                
                # 2. Encode ảnh sang latent (LDM)
                # Không cần gradient cho VAE
                with torch.no_grad():
                    # (B, 3, 32, 32) -> (B, 4, 4, 4)
                    mu, logvar = self.vae.encode(imgs)
                    # Dùng reparameterization để thêm noise, giúp VAE và UNet khớp nhau
                    z = self.vae.reparameterize(mu, logvar) 
                
                # 3. Tính loss trên latent z (LDM)
                # diffusion.loss_fn này đã được tối ưu (xem file Conditioning.py)
                loss = self.diffusion.loss_fn(z, caps_device) # <-- Tính loss trên z, không phải imgs

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {self.global_step}. Skipping batch.")
                self.optimizer.zero_grad()
                continue
                
            # 4. Backward với GradScaler (AMP)
            self.scaler.scale(loss).backward()
            
            # Optional: unscale trước khi clip
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.update_ema() # Cập nhật EMA
            
            # --- (Kết thúc khối tối ưu) ---

            if self.global_step % 50 == 0: # Log loss thường xuyên hơn
                 self.logger.log_losses({"train_loss": loss.item()}, self.global_step)

            # --- Sửa Sampling/Logging (dùng VAE.decode) ---
            if self.global_step % 1000 == 0 and self.evaluator is not None:
                print(f"Step {self.global_step}: Logging samples and evaluating...")
                self.model.eval() # Dùng model train (hoặc ema_model)
                with torch.no_grad():
                    sample_captions = list(caps_device[:4])
                    
                    if sample_captions:
                        _, context = self.diffusion.text_embedder.encode(sample_captions)
                        context = context.to(self.device)
                        
                        # Sampler trả về latents z
                        # (sampler.sample này đã được tối ưu, xem Guidance_Sampling.py)
                        z_samples = self.sampler.sample(batch_size=context.shape[0],
                                                        model=self.ema_model,
                                                        context=context)
                        
                        # Giải mã latents z -> ảnh pixels
                        samples_pixel = self.vae.decode(z_samples)
                        # Chuyển về [0, 1] để log
                        samples_pixel_log = (samples_pixel + 1.0) / 2.0
                        
                        self.logger.log_images(samples_pixel_log, self.global_step, name="generated_samples")
                        
                        # Đánh giá trên ảnh pixel
                        results = self.evaluator.evaluate(samples_pixel_log)
                        self.logger.log_metrics(results, self.global_step)
                    
                self.model.train() # Quay lại mode train
            
            total_loss += loss.item()
            self.global_step += 1
            
        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch} average loss: {avg_loss:.5f}")
        return avg_loss


    def save_checkpoint(self, path="checkpoint.pt", epoch=None):
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "step": self.global_step,
            "epoch": epoch if epoch is not None else 0
        }, path)

    def load_checkpoint(self, path="checkpoint.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ema_model.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["opt"])
        self.global_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Loaded checkpoint from {path}, resume at epoch {start_epoch}")
        return start_epoch

def benchmark_text_encoder(text_embedder, batch_size=8, seq_len=77, n_iters=50, device=None):
    """
    Đo hiệu suất CLIPTextEmbedder bằng cách tính tokens/second.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dummy_texts = ["a cat"] * batch_size

    # Warm-up (loại bỏ overhead lần đầu)
    for _ in range(5):
        _ = text_embedder.encode(dummy_texts)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    total_tokens = 0
    for _ in range(n_iters):
        _, seq_emb = text_embedder.encode(dummy_texts, max_length=seq_len)
        total_tokens += batch_size * seq_emb.shape[1]  # B * L

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    tokens_per_sec = total_tokens / elapsed
    print(f"ext encoder throughput: {tokens_per_sec:.2f} tokens/s "
          f"(batch={batch_size}, seq_len={seq_len}, iters={n_iters})")
    return tokens_per_sec

def benchmark_iterations(model, diffusion, batch_size=8, image_size=32, n_iters=50, device=None):
    """
    Đo số iterations (forward pass) trên UNet+diffusion mỗi giây.
    """
    import time
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fake input
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
    _, context = diffusion.text_embedder.encode(["benchmark"] * batch_size)

    # Warm-up
    for _ in range(5):
        _ = diffusion.model(x, t, context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_iters):
        _ = diffusion.model(x, t, context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    it_per_sec = n_iters / elapsed
    print(f"Diffusion throughput: {it_per_sec:.2f} iterations/s "
          f"(batch={batch_size}, image_size={image_size}, iters={n_iters})")
    return it_per_sec


def benchmark_images(sampler, model, text_embedder, batch_size=4, prompt="cat", n_iters=10, device=None):
    """
    Đo số images/s khi sampling với sampler (DDIM/DPMSolver...).
    """
    import time
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, context = text_embedder.encode([prompt] * batch_size)

    # Warm-up
    with torch.no_grad():
        _ = sampler.sample(batch_size=batch_size, model=model, context=context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(n_iters):
            _ = sampler.sample(batch_size=batch_size, model=model, context=context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    img_per_sec = (batch_size * n_iters) / elapsed
    print(f"Sampling throughput: {img_per_sec:.2f} images/s "
          f"(batch={batch_size}, n_iters={n_iters})")
    return img_per_sec