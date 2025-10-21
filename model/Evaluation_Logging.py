import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

# Metrics
from scipy import linalg
import clip
from PIL import Image
from torchvision.models import inception_v3
import torch.nn.functional as F

class Metrics:
    def __init__(self, device="cuda"):
        self.device = device
        # Load InceptionV3 for FID
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval()
        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def get_inception_features(self, images: torch.Tensor) -> torch.Tensor:
    # images in [0,1], shape (B,3,H,W)
        if images.shape[-1] < 75:
            images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        feats = self.inception(images.to(self.device))
        return feats

    def calculate_fid(self, real_images, fake_images):
        real_feats = self.get_inception_features(real_images)
        gen_feats  = self.get_inception_features(fake_images)

        real_feats = real_feats.detach().cpu().numpy()
        gen_feats = gen_feats.detach().cpu().numpy()

        mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
        mu2, sigma2 = np.mean(gen_feats, axis=0), np.cov(gen_feats, rowvar=False)

        diff = mu1 - mu2
        eps = 1e-6
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def calculate_clip_score(self, images: torch.Tensor, prompts: list[str]):
        # Convert images to PIL and preprocess
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
        clip_inputs = torch.stack([self.clip_preprocess(img) for img in pil_images]).to(self.device)

        with torch.no_grad():
            img_features = self.clip_model.encode_image(clip_inputs)
            text_tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)

        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (img_features @ text_features.T).diag().mean().item()
        return similarity


# Logger
class TrainingLogger:
    def __init__(self, logdir="logs", use_wandb=False, project="ddpm-training"):
        self.writer = SummaryWriter(logdir)
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project=project)

    def log_losses(self, losses: dict, step: int):
        for k, v in losses.items():
            self.writer.add_scalar(f"loss/{k}", v, step)
            if self.use_wandb:
                import wandb
                wandb.log({f"loss/{k}": v, "step": step})

    def log_images(self, images: torch.Tensor, step: int, name="samples"):
        grid = make_grid(images, nrow=4, normalize=True, value_range=(0,1))
        self.writer.add_image(name, grid, step)
        if self.use_wandb:
            import wandb
            wandb.log({name: [wandb.Image(grid.permute(1,2,0).cpu().numpy())], "step": step})

    def log_metrics(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"metric/{k}", v, step)
            if self.use_wandb:
                import wandb
                wandb.log({f"metric/{k}": v, "step": step})


# Evaluator
class Evaluator:
    def __init__(self, metrics: Metrics, prompts: list[str], real_images: torch.Tensor):
        self.metrics = metrics
        self.prompts = prompts
        self.real_images = real_images

    def evaluate(self, gen_images: torch.Tensor):
        fid = self.metrics.calculate_fid(self.real_images, gen_images)
        clip_score = self.metrics.calculate_clip_score(gen_images, self.prompts)
        return {"FID": fid, "CLIPScore": clip_score}


# Integration example
def training_loop(model, dataloader, optimizer, sampler, prompts, val_images, device="cuda"):
    metrics = Metrics(device)
    logger = TrainingLogger(use_wandb=True)
    evaluator = Evaluator(metrics, prompts, val_images)

    global_step = 0
    best_fid = 1e9
    for epoch in range(100):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()

            logger.log_losses({"train": loss.item()}, global_step)

            if global_step % 1000 == 0:
                # Sample images
                with torch.no_grad():
                    samples = sampler.sample(prompts, num_steps=50)
                logger.log_images(samples, global_step)

                # Evaluate
                results = evaluator.evaluate(samples)
                logger.log_metrics(results, global_step)

                # Save best
                if results["FID"] < best_fid:
                    best_fid = results["FID"]
                    torch.save(model.state_dict(), "best_model.pt")

            global_step += 1
