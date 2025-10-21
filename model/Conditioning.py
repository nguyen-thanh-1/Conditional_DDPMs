"""
Purpose:
- Extend Module 1 UNet/DDPM core to accept text conditioning via cross-attention.
- Integrate a CLIP text encoder (HuggingFace Transformers) as the default text encoder.
- Implement classifier-free guidance support (conditioning dropout during training and guidance during sampling).
- Provide training loop skeleton that consumes (image, caption) pairs.

Notes:
- This module builds on the UNet and GaussianDiffusion classes from Module1. You can import them
  if both files are in the same package. For convenience this file is written to be runnable
  after `module1_ddpm.py` is placed in the same folder.
- The code favors clarity and modifiability. For large-scale use, replace tokenizer/encoder
  with a frozen/pretrained CLIP text encoder checkpoint for best quality.

Requirements:
  pip install torch torchvision transformers sentencepiece

"""


from typing import List, Tuple, Optional
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# We import module1 components. Make sure module1_ddpm.py is in the same dir or installed.
from Diffusion_core import UNet, GaussianDiffusion, linear_beta_schedule, cosine_beta_schedule

# HuggingFace CLIP text encoder
from transformers import CLIPTokenizerFast, CLIPTextModel


# -------------------- Cross-Attention Block --------------------

class CrossAttention(nn.Module):
    """Multi-head cross-attention where queries come from image features and keys/values from text embeddings.
    """
    '''
    @input:
        - channels: number feature dimensions of queries (C)
        - context_dim: number of dimensions of vector context (D) (EX: D = 768 if using CLIP text embedding)
        - num_heads: number of head in multi-head attention
    @function:
        - initializing linear projects (Q/K/V) and parameters for multi-head cross-attention
        - scale: proportionalization dot product before softmax (stable gradient / numeric)
    '''
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5 # = (C / num_head)**(-1/2) = 1/sqrt(C(per_head))

        # projectors
        self.to_q = nn.Linear(channels, channels, bias=False) # project queries
        self.to_k = nn.Linear(context_dim, channels, bias=False) # project keys from context
        self.to_v = nn.Linear(context_dim, channels, bias=False) # project values from context
        self.to_out = nn.Linear(channels, channels) # project after grafting (combine) heads
        self.norm = nn.LayerNorm(channels) # normalization on channel dimensions of queries

    '''
    @input: 
        - x: (B, C, H, W) - image feature (queries)
        - context: (B, L, D) - sequence embedding from text (L = length of token text | D = context_dim)
    @output:
        - out: (B, C, H, W) - attention-enhanced image features 
    '''
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C) # Flatten spatial | goal: each position of pixel/patch is a token with embedding C
        x_flat = self.norm(x_flat)

        q = self.to_q(x_flat)  # (B, N, C)
        k = self.to_k(context)  # (B, L, C)
        v = self.to_v(context)  # (B, L, C)

        # reshape for heads
        q = q.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, Cph)
        k = k.view(B, k.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, L, Cph)
        v = v.view(B, v.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, L)
        attn = torch.softmax(attn, dim=-1) # attention weight for each query token i over context tokens j
        out = torch.matmul(attn, v)  # (B, heads, N, Cph) | Weighted sum values -> out per head
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C) | concatenate heads

        out = self.to_out(out)  # (B, N, C) #final linear projection
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out
    # summary formula: out(i) = W(o) {concat(h)(SUM(L;j=1)softmax( Q(i)K(i)^T / sqrt(C(ph)) )V(j) )}
    
    '''
    some ways alter for attn: flash attention, chunking, sparse attention
    '''

# -------------------- ResidualBlock with Cross-Attention --------------------
'''
ResidualBlockWithCA: a residual convolution block has integration time-conditioning (through projection of time embedding) and optional cross-attention to mix text infor into image feature map
'''
class ResidualBlockWithCA(nn.Module):
    '''
    @input:
        - in_ch: number of input channels (C_in)
        - out_ch: number of output channels (C_out)
        - time_emb_dim: vector dimensions of time embedding (d_t)
        - text_emb_dim: vector dimensions of text/context embedding (d_text)
        - dropout: rate of dropout(p)
        - use_cross_attention: whether cross-attention is enabled or not
        - ca_heads: number of head for cross-attention when using
    @function:
        - establishing 2 convolution residual path (res, res2), project for time embedding
        - optional cross-attention to combine with text infor into image features
        - skip ensures output has correct out_ch to add residual 
    '''
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, text_emb_dim: int, dropout: float = 0.0,
                 use_cross_attention: bool = True, ca_heads: int = 8):
        super().__init__()
        self.res = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.res2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch)) # project time embedding -> out_ch
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity() # residual shortcut

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = CrossAttention(out_ch, text_emb_dim, num_heads=ca_heads)

    '''
    @input:
        - x: (B, in_ch (C), H, W) - feature map input
        - t_emb: (B, time_emb_dim) - time embedding for batch
        - context (optional): (B, L, text_tmb_dim) - text/context token embeddings (if using use_cross_attention=True)
    @output:
        - out: (B, out_ch, H, W) - feature map output of block
    '''
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.res(x) # h1 = Conv1(SiLU(GroupNorm(x)))
        # add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None] # h2[b,c,i,j] = h1[b,c,i,j] + t'[b,c]
        h = self.res2(h) # h3 = Conv2(Dropout(SiLU(GroupNorm(h2))))
        if self.use_cross_attention and context is not None: # optional
            '''
            h(attn) = CrossAttn(h3, context)
            h4 = h3 + h(attn)
            if not: h4 = h3
            '''
            h = h + self.cross_attn(h, context) 
        return h + self.skip(x) 


# -------------------- Conditional UNet --------------------

class ConditionalUNet(UNet):
    """
    ConditionalUNet will inherit from UNet but modifying all of the residualBlock by ResidualBlockWithCA. 
    Aim: UNet predict ϵθ​(xt,t,context) (noise) that has condition by text context
    """

    '''
    @input:
        - text_emb_dim: embedding dimension of text/context (D)
        - *args, **kwarges: forward to UNet.__init__. In kwargs expect keys: in_channels, base_ch, ch_mults, num_res_blocks, time_emb_dim (if not it will use default values)
    '''
    def __init__(self, text_emb_dim: int = 768, *args, **kwargs):
        # Reuse UNet init but we will rebuild parts that use ResidualBlock
        super().__init__(*args, **kwargs)
        # Reconstruct key modules to instances with cross-attention.
        # Note: For simplicity, we re-create the architecture here rather than monkey-patch.
        # A production refactor would modularize block factories.
        # We'll create a new smaller UNet here that mirrors shapes from base UNet.

        # Basic config
        in_channels = kwargs.get('in_channels', 3)
        base_ch = kwargs.get('base_ch', 64)
        ch_mults = kwargs.get('ch_mults', (1, 2, 4))
        num_res_blocks = kwargs.get('num_res_blocks', 2)
        time_emb_dim = kwargs.get('time_emb_dim', 256)

        # Recreate time_mlp with same signature to ensure compatibility
        self.time_mlp = nn.Sequential(self.time_mlp[0], self.time_mlp[1], self.time_mlp[2]) if hasattr(self, 'time_mlp') else None
        # We'll rebuild encoder/decoder using ResidualBlockWithCA
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        in_ch = base_ch
        self.downs = nn.ModuleList() # list encoder layers (ResidualBlockWithCA and downsample convs)
        self.skips_channels = [] # save channels to use to skip connection
        for mult in ch_mults:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlockWithCA(in_ch, out_ch, time_emb_dim, text_emb_dim))
                in_ch = out_ch
                self.skips_channels.append(in_ch)
            self.downs.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))

        # block bottlenecks (using ResidualBlockWithCA)
        self.mid_block1 = ResidualBlockWithCA(in_ch, in_ch, time_emb_dim, text_emb_dim)
        self.mid_attn = nn.Identity()  # keep earlier attention if desired
        self.mid_block2 = ResidualBlockWithCA(in_ch, in_ch, time_emb_dim, text_emb_dim)

        # up blocks
        self.ups = nn.ModuleList()
        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.ups.append(nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)) # upsample spatial dims (H->2H, W->2W) and keep number of channels in_ch
            for _ in range(num_res_blocks):
                skip_ch = self.skips_channels.pop()
                self.ups.append(ResidualBlockWithCA(in_ch + skip_ch, out_ch, time_emb_dim, text_emb_dim)) # channel = in_ch + skip_ch, then passing to ResidualBlockWithCA(in_ch + skip_ch -> out_ch)
                in_ch = out_ch

        #output layers
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1) # mapping back to number of iamge channel input

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(t) if self.time_mlp is not None else None
        hs = [] # stack order LIFO
        h = self.init_conv(x) # init_conv maps input channels -> base_ch

        for layer in self.downs: # encoder
            if isinstance(layer, ResidualBlockWithCA):
                h = layer(h, t_emb, context)
                hs.append(h) # skip links
            else:
                h = layer(h) # reduce spatial dims H -> H/2
        # creating ResidualBlockWithCA at bottleneck; each accepts context and time embedding
        h = self.mid_block1(h, t_emb, context)
        h = self.mid_attn(h) # currently identity
        h = self.mid_block2(h, t_emb, context)

        for layer in self.ups: # decoder
            if isinstance(layer, nn.ConvTranspose2d): # doubles spatiaal dims (2H,2W) - matches skip spatial dims popped from hs
                h = layer(h) 
            else:
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb, context)
        '''
        shape:
            - before upsample: (B, in_ch, H, W)
            - after ConvTranspose2d: (B, in_ch, 2h, 2W)
            - skip from hs: (B, skip_ch, 2H, 2W)
            - concat -> (B, in_ch + skip_ch, 2H, 2W) -> ResidualBlockWithCA produces (B, out_ch, H, W)
        '''
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        # out_conv maps in_ch back to in_channels (image channels)- so final h shape (B, in_channels, H, W)
        # this h is the predicted ε
        return h
    '''
    summary formula:
        - time embedding: t(emb) = time_mlp(t)
        - each ResidualBlockWithCA: y = Skip(x) + (Conv2( SiLU( GN( Conv1( SiLU( NG(x) ) ) + W(t)t(emb) ) ) + CrossAttn(.,context))) (if cross-attn is turned on by context != None)
        - Downsample conv stride=2: h' = Conv(stride2)(h)
        - Decoder concat & residual: h' = concat(h(up), h(skip)) (channel dim)
                                     h' = ResidualBlockWithCA(h(cat), t(emb, context))
        - final output: ε(theta)(x, t, context) = out_conv(SiLU(GroupNorm( h(final) )))
    '''

# -------------------- Text Encoder Wrapper --------------------
'''
- Wrapper download tokenizer + CLIP text encoder (HuggingFace)
- Encode one batch of prompts into input_ids and sequence token embeddings (seq_emb) to use for cross-attention in ConditionalUNet
'''
class CLIPTextEmbedder(nn.Module):
    '''
    @input:
        - pretrained: name model/tokenizer HF
        - device: optional
    @function:
        - Downloading tokenizer and text encoder pretrained, preparing for encode text into token IDs and embedding that UNet use as the context for cross-attention
        - using freezed model to save memory and keep weights constant (no fune-tune following by default)
    '''
    def __init__(self, 
                 pretrained: str = "openai/clip-vit-large-patch14", 
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Load tokenizer + model
        self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained)
        self.model = CLIPTextModel.from_pretrained(pretrained).to(self.device)
        self.text_emb_dim = self.model.config.hidden_size  # = 768 for CLIP-L/14

        # freeze text encoder 
        self.model.eval() # set inference mode (turn off dropout) 
        for p in self.model.parameters():
            p.requires_grad = False
    '''
    @input:
        - texts: batch of prompts
        - max_length: maximum length of sequence
    @output:
        - input_ids: (B, L) (L: max length in batch after padding, <= max_length)
        - seq_emb: (B, L, D) (D = self.text_emb_dim)
    @function:
        - list text will be converted token IDs (padding/truncation) and running encoder to take sequence token embeddings
        - These embeddings will be contexts for corss-attention (B, L, D)
    '''
    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:

        toks = self.tokenizer(
            texts, 
            padding="longest", 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        input_ids = toks["input_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)

        out = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=False
        )
        seq_emb = out.last_hidden_state  # [B, L, D]
        return input_ids, seq_emb


# -------------------- Classifier-free Guidance helpers --------------------
'''
- Used to implement a simple form of classifier-free guidance training
- Aim: during training, apart of batch will be trained with uncondition (untext) to enable classifier-free guidance when sampling
'''

'''
@input:
    - seq_emb: sequence embeddings (B=batch sise, L=sequence length, D=text embedding dim)
    - keep_prob: keep probability context
@output:
    optional
        - None: when deciding drop all context for batch (unconditioned)
        - seq_emb (origin): when keep context (conditioned)
'''
def maybe_mask_context(seq_emb: torch.Tensor, keep_prob: float = 0.9) -> Optional[torch.Tensor]:
  
    if torch.rand(1).item() < (1.0 - keep_prob):
        return None
    return seq_emb


# -------------------- Conditional Diffusion wrapper --------------------
'''
- conditionalGaussianDiffusion has additional capability about text conditioning by using text_embedding (CLIP)
- support classifier-freee guidance in sample_with_guidance
- loss_fn: implement dropout condition per-sample
'''
class ConditionalGaussianDiffusion(GaussianDiffusion):
    '''
    @input:
        - model: Conditional UNet
        - text_embedder: instance CLIPTextEmbedder - using to encode prompts
        - **kwargs: forward to parameters of GaussianDiffusion.__init__(...)
    '''
    def __init__(self, model: nn.Module, text_embedder: CLIPTextEmbedder, **kwargs):
        super().__init__(model=model, **kwargs)
        self.text_embedder = text_embedder
        
        # --- TỐI ƯU CFG-TRAIN ---
        # Tạo và cache null_context một lần
        print("Caching null context for training...")
        with torch.no_grad():
             # (1, L, D)
            _, self.null_context = self.text_embedder.encode([""])
        self.null_context = self.null_context.to(self.device)
        print(f"Null context shape: {self.null_context.shape}")
        # --- (Kết thúc) ---

    '''
    @input:
        - x_t: (B, C, H, W)
        - t: (B,)
        - context: (B, L, D) or None
    @output:
        - mean_pred: (B, C, H, W) - mean of approximated reverse distribution p(theta)(xt-1|xt)
        - var: (B, 1, 1, 1) broadcastable - posterior variance beta^t (from scheduler)
    @function:
        - using model to predict noise eps(theta)(xt, t, context) -> recreate x^0
        - Calculate mean & variance of p(xt-1|xt) following DDPM formula
    '''
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        eps_pred = self.model(x_t, t, context)
        x0_pred = (self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t -
                   self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * eps_pred)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        mean_pred = (
            self.posterior_mean_coeff1(t).view(-1, 1, 1, 1) * x0_pred +
            self.posterior_mean_coeff2(t).view(-1, 1, 1, 1) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean_pred, var

    '''
    @output:
        - x{t-1}: (B< C< H, W - sampled previous step
    @function: 
        - calculate mean, var --> take sample x(t-1) ~ N(mean, var)
    @formular:
        x(t-1) = mi + sqrt(var).z , z ~ N(0,I)
    '''
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean, var = self.p_mean_variance(x_t, t, context)
        if t[0] == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise #

    '''
    @input:
        - batch_size: number of image that is generated
        - context_texts: list prompts. Length should match batch_size
        - guidance_scale: classifier-free guidance weight w 
    @output:
        - x: (B, C, H, W) - sampled images in final step
    @function:
        - Generating batch image with classifier-free guidance: at each step compute eps_cond and eps_uncond
        - combine them into guided epsilon, and perform ancestral sampling step
    '''
    def sample_with_guidance(self, batch_size: int, context_texts: list, guidance_scale: float = 5.0, device: Optional[torch.device] = None):
        device = device or self.device
        # encode texts (B, L, D)
        _, context = self.text_embedder.encode(context_texts)
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device) #initialize
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            # classifier-free guidance: run conditioned and unconditioned model and combine

            # conditioned eps
            eps_cond = self.model(x, t, context)
            # unconditioned eps (context dropped)
            eps_uncond = self.model(x, t, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # compute x_{t-1} from eps
            x0_pred = (self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x -
                       self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * eps)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            coef1 = self.posterior_mean_coeff1(t).view(-1, 1, 1, 1)
            coef2 = self.posterior_mean_coeff2(t).view(-1, 1, 1, 1)
            mean = coef1 * x0_pred + coef2 * x
            var = self.posterior_variance[t].view(-1, 1, 1, 1)

            if i == 0:
                x = mean
            else:
                x = mean + torch.sqrt(var) * torch.randn_like(x)
        return x
    '''
    @input:
        - x_start: (B, C, H, W) - clean training images (normalized consistent with model)
        - captions: length B - text prompts for each training example
        - cond_drop_prob: probability to drop conditioning for each element (per-element dropout) during training
    @output:
        - scalar tensor - MSE loss
    @function:
        - creating x_t
        - compute eps predictions and MSE with ground-true noise
    '''
    '''
    Sửa loss_fn để dùng null_context (Tối ưu #1)
    '''
    def loss_fn(self, x_start: torch.Tensor, captions: list, cond_drop_prob: float = 0.1) -> torch.Tensor:
        B = x_start.shape[0]
        # x_start bây giờ là latent z (B, 4, 4, 4)
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise) # (B, 4, 4, 4)

        # encode captions
        with torch.no_grad(): # Không cần grad cho text encoder
            _, seq_emb = self.text_embedder.encode(captions) # (B, L, D)

        # --- TỐI ƯU CFG-TRAIN (Masking) ---
        
        # Tạo mask (B,) -> (B, 1, 1)
        # 1.0 = giữ context, 0.0 = drop (dùng null_context)
        keep_mask = (torch.rand(B, device=self.device) > cond_drop_prob).float()
        mask = keep_mask.view(B, 1, 1) # (B, 1, 1)

        # Lặp null_context cho vừa batch (B, L, D)
        null_ctx_batch = self.null_context.repeat(B, 1, 1)

        # Trộn: (B, L, D) * (B, 1, 1) + (B, L, D) * (B, 1, 1)
        # Nếu mask=1, giữ seq_emb. Nếu mask=0, dùng null_ctx_batch.
        context = seq_emb * mask + null_ctx_batch * (1.0 - mask)
        
        # Chạy model MỘT LẦN
        eps_pred = self.model(x_t, t, context)
        
        # --- (Kết thúc) ---

        return F.mse_loss(eps_pred, noise)


# -------------------- Example dataset for (image, caption) --------------------

class DummyImageCaptionDataset(Dataset):
    """Small dummy dataset to test conditioning pipeline. Replace with COCO/LAION loader in production."""
    def __init__(self, image_size: int = 32, num_samples: int = 1000):
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.num = num_samples
    def __len__(self):
        
        return self.num

    def __getitem__(self, idx):
        # return a random noise image plus a simple caption (toy example)
        img = torch.randn(3, 32, 32) # original (3, 32, 32)
        img = torch.clamp(img, -1.0, 1.0)
        caption = f"A random pattern {idx % 10}"
        return img, caption


# -------------------- Training loop for Module 2 --------------------

def train_module2(model: nn.Module, diffusion: ConditionalGaussianDiffusion, dataset: Dataset, epochs: int = 5,
                  batch_size: int = 16, lr: float = 2e-4, save_path: str = './checkpoints_module2'):
    device = diffusion.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (x, captions) in enumerate(dataloader):
            x = x.to(device)
            loss = diffusion.loss_fn(x, captions, cond_drop_prob=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i+1}/{len(dataloader)} | Loss: {running_loss / 50:.5f}")
                running_loss = 0.0
        # checkpoint
        ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'epoch': epoch}
        torch.save(ckpt, os.path.join(save_path, f'ckpt_epoch_{epoch}.pt'))
        # sample with guidance from a few prompts
        model.eval()
        with torch.no_grad():
            prompts = ["A colorful abstract pattern", "A simple line drawing", "A noisy texture"]
            samples = diffusion.sample_with_guidance(batch_size=3, context_texts=prompts, guidance_scale=4.0)
            grid = (samples + 1.0) / 2.0
            utils.save_image(grid, os.path.join(save_path, f'samples_epoch_{epoch}.png'), nrow=3)
        model.train()
    print('Module 2 training finished')
