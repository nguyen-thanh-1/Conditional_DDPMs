"""
Purpose:
- Provide a lightweight VAE-like autoencoder (encoder + decoder + quantized/gaussian latent options)
  suitable for training a diffusion model in latent space (a simple approximation of LDM pipeline).
- Utilities to train the autoencoder (reconstruction + optional KL), save/load checkpoints.
- Wrapper that plugs a pretrained UNet diffusion model to run diffusion in latent space: encode images -> run diffusion -> decode latents -> images.
- Sampling and guidance hooks compatible with Module2/Module3 samplers.

Notes:
- This implementation focuses on clarity and educational value rather than matching the full Stable Diffusion codebase.
- For scaling to high-resolution (512×512), replace this VAE with a high-capacity AutoencoderKL and increase latent capacity.
"""

from typing import Optional, Tuple
import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# Reuse UNet from module1 or ConditionalUNet from module2 as the diffusion model in latent space.
# from module1_ddpm import UNet
# from module2_conditioning import ConditionalUNet
from Conditioning import ConditionalGaussianDiffusion

# -------------------- Small Convolutional VAE (Encoder / Decoder) --------------------

class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_ch: int = 64, z_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(base_ch * 4, z_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        z = self.conv4(h)
        return z  # latent tensor (B, z_channels, H/8, W/8)


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int = 3, base_ch: int = 64, z_channels: int = 4):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(z_channels, base_ch * 4, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.conv4 = nn.Conv2d(base_ch, out_channels, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(z))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        x_recon = torch.tanh(self.conv4(h))  # output in [-1,1]
        return x_recon


class SimpleVAE(nn.Module):
    """A small VAE-style autoencoder with gaussian posterior (mu, logvar) per latent channel.
    Not exactly the AutoencoderKL from LDM, but sufficient to illustrate latent diffusion pipeline.
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 64, z_channels: int = 4):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, base_ch=base_ch, z_channels=z_channels * 2)
        # encoder outputs 2*z_channels to represent mu and logvar
        self.z_channels = z_channels
        self.decoder = ConvDecoder(out_channels=in_channels, base_ch=base_ch, z_channels=z_channels)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns mu, logvar
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# -------------------- Training utilities for VAE --------------------

@dataclass
class VAEConfig:
    lr: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    recon_weight: float = 1.0
    kl_weight: float = 0.01


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, recon_weight: float = 1.0, kl_weight: float = 0.01) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x)
    # KL divergence between N(mu, var) and N(0,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / (recon.shape[0] * recon.shape[1] * recon.shape[2] * recon.shape[3])
    return recon_weight * recon_loss + kl_weight * kl


def train_vae(vae: SimpleVAE, dataset: str = 'CIFAR10', image_size: int = 32, config: VAEConfig = VAEConfig(), save_path: str = './vae_ckpts'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=config.lr)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    if dataset == 'CIFAR10':
        ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise NotImplementedError
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(config.epochs):
        vae.train()
        running = 0.0
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar, config.recon_weight, config.kl_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{config.epochs} | Step {i+1}/{len(loader)} | Loss: {running/100:.5f}")
                running = 0.0
        # save checkpoint and sample reconstructions
        ckpt = {'vae_state_dict': vae.state_dict(), 'epoch': epoch}
        torch.save(ckpt, os.path.join(save_path, f'vae_epoch_{epoch}.pt'))
        # sample reconstructions
        vae.eval()
        with torch.no_grad():
            xs = next(iter(loader))[0][:16].to(device)
            recon, _, _ = vae(xs)
            grid = torch.cat([xs, recon], dim=0)
            grid = (grid + 1.0) / 2.0
            utils.save_image(grid, os.path.join(save_path, f'recon_epoch_{epoch}.png'), nrow=8)
    print('VAE training finished')


# -------------------- Latent Diffusion Wrapper -------------------- 

class LatentDiffusion:
    """Wrapper that runs diffusion in latent space. Assumes:
      - vae.encode(x) -> mu, logvar
      - vae.reparameterize(mu, logvar) -> z (B, z_ch, h, w)
      - vae.decode(z) -> x_recon
      - diffusion.model expects input with channels = z_channels and image_size = latent_size (h,w)
    """
    def __init__(self, vae: SimpleVAE, diffusion: ConditionalGaussianDiffusion, vae_scale_factor: int = 8):
        self.vae = vae
        self.d = diffusion
        self.device = diffusion.device
        self.vae_scale_factor = vae_scale_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # return latent z (no grad) in device of diffusion
        self.vae.to(self.device)
        x = x.to(self.device)
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterize(mu, logvar)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.vae.to(self.device)
        with torch.no_grad():
            x = self.vae.decode(z)
        return x

    def train_diffusion_step(self, x: torch.Tensor) -> torch.Tensor:
        # Example: given a batch of images x in [-1,1], encode to z and compute diffusion loss on z
        z_mu, z_logvar = self.vae.encode(x)
        z = self.vae.reparameterize(z_mu, z_logvar)
        # ensure shapes align with diffusion.image_size (latent H/W)
        # we assume diffusion.image_size == z.shape[-1]
        return self.d.loss_fn(z)

    def sample(self, batch_size: int, prompts: Optional[list] = None, guidance_scale: float = 4.0, sampler=None):
        # sample latents using provided sampler (e.g., Module3 DDIMSampler). sampler.sample should return latents in [-1,1]
        if sampler is None:
            # fallback to diffusion.sample
            z = self.d.sample(batch_size=batch_size, device=self.device)
        else:
            # sampler should accept model=diffusion.model and context if present
            z = sampler.sample(batch_size=batch_size, model=self.d.model, context=prompts)
        # decode
        x = self.decode(z)
        return x
