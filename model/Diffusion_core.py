"""
Features included:
- Noise schedules (linear, cosine)
- Forward q_sample (adding noise)
- UNet-like denoiser with Time Embedding
- Training loss (simple MSE on predicted noise)
- Sampling loop (ancestral DDPM sampling + optional DDIM stub)
- Checkpointing and basic training loop skeleton

"""

from typing import Optional, Tuple
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------- Utilities: Noise Schedules --------------------
'''
Noise schedule: Quy định cách nhiễu Beta(t) thay đổi theo timestep, nó sẽ giúp tạo ra dạy noise schedule để xây dựng các biên liên quan như alpha(t) = 1 - beta(t), alpha_mũ(t) = PI(t)(s=1)alpha(s)
'''
def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Linear schedule from beta_start to beta_end over timesteps."""
    return torch.linspace(beta_start, beta_end, timesteps)
'''
Đầu vào:
    timesteps: số bước khuếch tán T (DDPM gốc dùng T = 1000)
    beta_start: giá trị Beta(1), tức lượng nhiễu thêm ở bước đầu, thường rất it để ảnh ban đầu chưa bị phá nhiều
    beta_end: giá trị Beta(T), tức lượng nhiễu ở bước cuối cùng, thường lớn hơn để đảm bảo x(T)~N(0,I)

Đâu ra:
    torch.tensor có shape(timesteps,): chứa dãy Beta(t) từ t=1 -> t=T, được nội suy tuyến tính từ Beta_start -> Beta_end (công thức cho nội suy tuyến tính để tính Beta(t) = Beta_start + [(t-1) / (T-1)](Beta_end - Beta_start)

EX:  linear_beta_schedule(5, 0.1, 0.5) --> output: tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000])
'''

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (improved).
    Returns beta_t for t in 0..timesteps-1 as a torch.Tensor.
    """
    '''
    cosine schedule cải tiến hơn so với linear schedule 
    Alpha_mũ(t) giảm theo hàm cos^2, tức ban đầu giảm chậm, sau đó nhanh lên --> giữ lại nhiều thông tin ở giai đoạn đầu, training ổn hơn
    Nghiên cứu cho thấy cosine schedule giúp model tọa ảnh rỏ nét hơn vả chất lượng cao hơn
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) # tạo x = [0, 1,...,T]: tạo dãy giá trị t từ 1 đến T 
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] 
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999) # đảm bảo Beta(t) thuộc [10^-4, 0.999], tránh quá nhỏ hoặc quá lớn gây bất ổn số học
'''
Đầu vào: 
    timesteps: số bước diffusion T
    s: offset nhỏ để tránh cos^2 = 0 hoàn toàn, giúp ổn định training

Đầu ra: 
    betas: torch.Tensor với shape(timestep,)
    Đây là dãy Beta(t) được suy ra từ cosine schedule  
'''

# -------------------- Time Embedding --------------------
'''
timestep embedding + x(t) sẽ là đầu vào cho UNet 
embedding này đưa qua MLP và trông vào residula blocks hoặc attention blocks
Nhờ đó mô hình biết được đang xử lý timestep nào
Nếu ko có embedding thì UNet không biết sự khác biệt giữa x10 và x500 là gì
'''
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps.
    Input: (batch,) long tensor of timesteps
    Output: (batch, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim # số chiều embedding muốn sinh ra

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None].float() * emb[None, :] # nhân t với vector tần số ( shape(B, haft_dim) )
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # láy sin, cos --> tạo embedding tuần hoàn. Shape(B, dim) hoặc shape(B, 2*half_dim)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1)) # nếu dim số lẻ, pad thêm 0 để giữ shape(B, dim)
        return emb  # (B, dim)
"""
- Hàm biến đổi để mã hóa giá trị thời gian rời rạc (timestep) thành vector nhiều chiều bằng các hàm sin, cos tuần hoàn với nhiều tần số khác nhau
- Phục vụ cho việc giúp mô hình hiểu timestep trong quá trinhg train 
"""
'''
input:
    t: timestep trong quá trình diffusion, mỗi phần tử là một giá trị t thuộc [0,T] ( shape(B,) )
output:
    tensor shape(B, dim)

Đây là timestep embedding vector mà mô hình sẽ dùng như một tín hiệu phụ thêm vào mạng UNet

EX: SinusoidalPosEmb(8)(torch.tensor([1, 2])) --> output: tensor shape (2, 8)
'''
# -------------------- Basic Blocks for UNet --------------------

class ResidualBlock(nn.Module):
    '''
    - Residual Block
    Mục đích: xử lý đặc trung ảnh (convolution) và nhận biết timestep embedding 
    Nhiệm vụ: Trích xuất đặc trưng, thêm thông tin thời gian t vào dòng xử lý, dùng réidual connection (skip) để giữ lại gradient ổn định
    '''
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        # chuẩn hóa và convolution
        self.norm1 = nn.GroupNorm(8, in_ch) # Chia kênh (channel) thành 8 nhóm, chuẩn hóa để ổn định khi batch nhỏ
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1) # giữ kích thức không gian (H x W), thay đổi số kênh từ in_ch -> out_ch
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Time embedding projection
        '''
        - time_mpl: một MLP (multi-layer pểcption) rất nhỏ
                chức năng: biến đổi time embedding (vector sinusoidal) sang không gian đặc trung phù hợp để + vào feature map trong UNet
        - nin=shortcut: nếu in_ch != out_ch dùng conv 1x1 để đổi số kênh --> match out_ch / nếu ko thì giữ nguyên
            --> đây chính là residual skip connection
        - Dropout: giúp regularization, tránh overfiting, tăng đa dạng
        '''
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch)) # từ vector t(emb) (output từ SinusoidalPosEmb + MLP), chiếu sang không gian cùng số kênh out_ch
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity() # đảm bảo match kích thước khi cộng
        self.dropout = nn.Dropout(dropout) # Giúp regularization và tăng khả năng học hiểu mạnh hơn

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # norm -> activation -> conv1
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        # Norm -> activation -> dropout -> conv2
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.nin_shortcut(x) # skip connection
    


'''
Công thức: cho input feature x và time embedding e(t):
    h1 = Conv1(SiLU(GroupNorm(x)))
    h2 = h1 + W(t)e(t)
    h3 = Conv2(Dropout(SiLU(GroupNorm(h2))))
    y = h3 + Shortcut(x)
'''

class AttentionBlock(nn.Module):
    """Simple self-attention block over channels (like SAGAN/BigGAN style but adapted).
    Input shape: (B, C, H, W) (batch, channels, height, width) -> flatten to (B, H*W, C) for attention.
    """
    '''
    AttentionBlock: dùng cơ chế self-attention vòa U-Net trong conditional DDPMs
        self-attention: cho phép mỗi pixel (token) nhìn toàn ảnh và học được mối quan hệ giữa các vùng ảnh xa nhau
        vai trò của khối: 
            + giúp mô hình kết hợp với ngữ nghĩa từ text + cấu trúc không gian ảnh 
            + Bổ sung ngữ cảnh toàn cục cho pipeline convolution, làm ảnh sinh ra nhất quán và nghĩa rõ hơn
    '''
    '''
    @input: 
        x ∈ R(BxCxHxW)
            B: batch size
            C: số channels (features từ CNN / input khởi tạo)
            H, W: chiều ảnh 
        num_head: số lượng head trong multi-head (chia channels thành nhiều phần nhỏ) self-attention. Mỗi head học mối quan hệ không gian khác nhau, rồi kết quả được ghép lại.
    @output:
        Tensor cùng shape(B,C,H,W) nhưng được thêm ngữ cảnh toàn ảnh (self-attention)
        Nhờ skip connection (x_in + out), block giữ thông tin gốc + thông tin attention
    '''
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)
        x = x.view(B, C, H * W)  # (B, C, N) # reshape(B,C, N) với N = HxW

        # Calculate Query, Key, Value
        q = self.q(x)  # (B, C, N)
        k = self.k(x)
        v = self.v(x)

        # reshape for multihead: (B, heads, C_per_head, N) 
        C_per_head = C // self.num_heads
        q = q.view(B, self.num_heads, C_per_head, -1) # shape(B, h:num_head, C(h): C_per_head, N: HxW)
        k = k.view(B, self.num_heads, C_per_head, -1)
        v = v.view(B, self.num_heads, C_per_head, -1)

        # scaled dot-product attention
        # attn: attention weight | đo độ similarity giữa pixel
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * (1.0 / math.sqrt(C_per_head)) # tính attention score giữa Query và Key
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(B, C, -1) # view: thay đổi kích thước tensor nhưng vẫn giữ nguyên dữ liệu | -1: Pytorch sẽ tự động tính số chiều còn lại sao cho phù hợp, nhưng ở đây -1 chính là H*W
        out = self.pproolj_out(out)
        out = out.view(B, C, H, W)

        return x_in + out


# -------------------- UNet Denoiser --------------------

class UNet(nn.Module):
    """Simplified UNet with down/up blocks, residual blocks and attention in middle.

    Input: noisy image x_t, timestep t
    Output: predicted noise eps_theta(x_t, t)

    Shapes (example for 3-channel RGB images):
        x: (B, 3, H, W)
        t: (B,) long
    
    Mục đích của UNet: mạng dự đoán nhiễu trong conditional DDPMs, kết hợp local + global features để sinh ảnh rõ, hợp ngữ cảnh
    self-attention: Cơ chế giúp model hiểu các phần tử trong cùng dữ liệu
                    trong trường hợp này là ảnh thì mỗi pixel sẽ không chỉ lưu thông tin của nó mà còn biết về những pixel khác bằng việc tính độ tương đồng của một pixel với những pixel khác bằng hmaf Softmax
    skip-connections: là quá trình nối tắt đặc trưng giữa encoder sang decoder trong UNet
                    Khi encoder (downsampling nhiều lần), mô hình sẽ có được ngữ cảnh thông qua đặc trưng nhưng lại mất chi tiết cục bộ. Skip-connetions sẽ truyền trực tiếp feature maps từ encoder sang decoder ở cùng mức phân giải
   """
    def __init__(self, in_channels: int = 3, base_ch: int = 64, ch_mults=(1, 2, 4, 8),
                 num_res_blocks: int = 2, time_emb_dim: int = 256):
        super().__init__()
        # time embedfing: biến timestep t thành vector nhúng giàu thông tin, truyền vào residual blocks giúp nhận biết đang xử lý ở bước nào trong diffusion process
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))

        # initial conv
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # down blocks (encoder): trích xuất đặc trưng toàn cục từ ảnh qua nhiều tầng giảm kích thức không gian (downsampling)
        in_ch = base_ch
        self.downs = nn.ModuleList()
        self.skips_channels = []
        for mult in ch_mults:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
                self.skips_channels.append(in_ch)
            # downsample
            self.downs.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))

        # middle (bottleneck): kết hợp resuidual + attention -> học cả đặc trưng cục bộ (CNN) và toàn cục (attention).
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # up blocks (decoder): khôi phục ảnh về kích thước gốc, dùng skip-connection để ghép đặc trưng encoder và giữ chi tiết
        self.ups = nn.ModuleList()
        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            # upsample
            self.ups.append(nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                # skip connection will concatenate, so input channels = in_ch + skip_ch
                skip_ch = self.skips_channels.pop()
                self.ups.append(ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim))
                in_ch = out_ch

        # final convs: đưa output về cùng số kênh với ảnh input (3 kênh RGB)
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return predicted noise (epsilon) with same shape as x."""
        # x: (B, C, H, W). t: (B,)
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        hs = []

        h = self.init_conv(x)
        # Down
        i = 0
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
                hs.append(h)
            else:  # downsample conv
                h = layer(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Up
        for layer in self.ups:
            if isinstance(layer, nn.ConvTranspose2d):
                h = layer(h)
            else:  # ResidualBlock expecting concatenated skip
                skip = hs.pop()
                # concat along channels
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h 
    # output: Nhiễu dự đoán

'''
Quy trình UNet: Khử nhiễu ảnh
        Input: ảnh nhiễu tại timestep x(t). Thông tin timestep t sẽ giúp mô hình biết được đang xử lý ở bước nào thông qua encode bằng positional embedding
        Encode (Downsampling path): ảnh sẽ qua Conv + norm + activation / Mỗi tầng sẽ giảm kích thước không gian (H*W) và tăng kênh đặc trưng C / đầu ra sẽ giàu thông tin ngữ nghĩa nhưng mất chi tiết cục bộ
        Bottleneck: Tầng sâu nhất, nén nhất (nới chứa nhiều thông tin ngữ nghĩa nhất )

'''

'''
Encode ảnh bằng downsampleing path
Them thông tin thời gian vào từng block
Capture global context ở middle attention block
Decode lại bằng upsampling path + skip connections
--> dự đoán nhiễu ϵ(theta)(x(t),t)
'''
# -------------------- Gaussian Diffusion Utilities --------------------

@dataclass
class DiffusionHyperparams:
    timesteps: int = 1000
    beta_schedule: str = 'linear'  # 'linear' or 'cosine'

# có thể đổi giữa linear và cosine để đánh giá hiệu suât
"""
@input:
    - model: UNet (hoặc model) dự đoán nhiễu ϵθ​(xt,t)
    - image_size: H=W=image_size
    - channels: số channel ảnh
    - timestep: số bước T
    - beta_schedual: linear or cosine
"""
class GaussianDiffusion:
    def __init__(self, model: nn.Module, image_size: int = 32, channels: int = 3,
                 timesteps: int = 1000, beta_schedule: str = 'linear', device: Optional[torch.device] = None):
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # betas: kết quả tensor shape (T,) chưa beta(t) cho t = 0..T-1
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unknown beta schedule')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # alpha^ = ∏(t, s=0)alpha(s)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # register as buffers so they move with .to(device) and saved in checkpoints
        self.register_buffer = lambda name, val: setattr(self, name, val.to(self.device))

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    '''
    q_sample: láy x_0 và thêm nhiễu Gaussian theo công thức closed-form của forward process
    @input:
        - x_start: ảnh sạch x0 shape(B, C, H, W)
        - t: timestep
        - noise: nhiễu, None thì tạo N(0,I)
    @output:
        - x_t: ảnh tại bước t theo q, tức x ~ q(xt|x0)
    '''
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) by adding noise according to schedule.
        x_start: (B, C, H, W)
        t: (B,) long (0..T-1)
        returns x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    '''
    @input:
        - x_t: (B, C, H, W) - ảnh noise tại bước t
        - t: (B,) 
        - x0_pred: (B, C, H, W) - ước lượng x^0
    @output:
        - esp (predicted): (B, C, H, W) - ước lượng noise ϵ^
    '''
    def predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        return (x_t - self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0_pred) / self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    '''
    @input:
        - x_t: (B, C, H, W) - noisy image at t
        - t: (B,)
    @output:
        - mean_pred: (B, C, H, W) - mean of approxomate posterior (trung bình của hậu nghiệm sắp xỉ) pθ(xt-1|xt) (dùng để láy sample)
        - var: (B, 1, 1, 1) broadcastable - variance (posterior variance) at t
    function: Using model to predict noise esp(theta), then estimating x^0. Finally, calculating mean & var of distribution p(xt-1|xt) following posterior formula (DDPM formula)
    '''
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given x_t, predict mean and variance for p(x_{t-1} | x_t)
        Model outputs predicted noise eps_theta(x_t, t)
        """
        eps_pred = self.model(x_t, t)
        # predict x0
        x0_pred = (self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t -
                   self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * eps_pred)

        # clamp x0 for numerical stability: giới hạn giá trị về [-1;1]
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        mean_pred = (
            self.posterior_mean_coeff1(t).view(-1, 1, 1, 1) * x0_pred +
            self.posterior_mean_coeff2(t).view(-1, 1, 1, 1) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean_pred, var

    def posterior_mean_coeff1(self, t: torch.Tensor) -> torch.Tensor:
        # coef for x0 in posterior mean
        return (self.betas[t] * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t]))

    def posterior_mean_coeff2(self, t: torch.Tensor) -> torch.Tensor:
        # coef for x_t in posterior mean
        return ((1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(self.alphas[t]) / (1.0 - self.alphas_cumprod[t]))

    '''
    @input:
        - x_t: (B, C, H, W) - current noisy image at step t
        - t: (B,) - timestep
    @output:
        - x(t-1): (B, C, H, W) - sample from q(xt-1|t) using predicted mean & var 
    @function: 
        performing one step sampling: predicting mean & var by p_mean_variance -> return mean (if t==0) else mean + sqrt(var)*z (if t > 0)
    '''
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t) using model predictions.
        x_t: (B, C, H, W)
        t: (B,) long
        returns x_{t-1}
        """
        mean, var = self.p_mean_variance(x_t, t)
        if t[0] == 0:
            # if t==0, we return mean (no noise)
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    '''
    @input:
        - batch_size: number of image that needing to generate
        - device: device to generate
    @output:
        - x_0_sampled: shape(batch_size, channels, image_size, image_size) - final generated image
    @function:
        - performing ancestral sampling full chain progress (which means: running all of timestep T from noise to image, each step also has sampling (add random noise and noise reduction))
    '''
    def sample(self, batch_size: int = 16, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x

    '''
    @input:
        - x_start: (B, C, H, W) - batch clean image (train images), normalized
    @output:
        - loss: scalar tensor - MSE loss (Mean Squared Error) between noise ground-truth and eps_pred returned by model
    @function:
        calculating main training loss (L-simple) for diffution
    '''
    def loss_fn(self, x_start: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch x_start.
        Steps:
          - sample t ~ Uniform({0..T-1})
          - sample noise
          - compute x_t = q_sample(x_start, t, noise)
          - predict eps_pred = model(x_t, t)
          - loss = MSE(noise, eps_pred)
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        eps_pred = self.model(x_t, t)
        return F.mse_loss(eps_pred, noise)


# -------------------- Training Loop Skeleton --------------------

def get_dataloader(dataset_name: str = 'CIFAR10', image_size: int = 32, batch_size: int = 32, data_root: str = './data') -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # in [0,1]
        transforms.Normalize([0.5], [0.5])  # scale to [-1,1]
    ])

    if dataset_name == 'CIFAR10':
        ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    else:
        raise NotImplementedError

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def train(model: nn.Module, diffusion: GaussianDiffusion, epochs: int = 10, batch_size: int = 32, lr: float = 2e-4,
          save_path: str = './checkpoints', dataset_name: str = 'CIFAR10'):
    device = diffusion.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    dataloader = get_dataloader(dataset_name=dataset_name, image_size=diffusion.image_size, batch_size=batch_size)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            loss = diffusion.loss_fn(x)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i+1}/{len(dataloader)} | Loss: {running_loss / 100:.5f}")
                running_loss = 0.0

        # Save checkpoint and sample
        ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch,
            'timesteps': diffusion.timesteps,
        }
        torch.save(ckpt, os.path.join(save_path, f'ckpt_epoch_{epoch}.pt'))

        # sample some images for visual monitoring (small batch)
        model.eval()
        with torch.no_grad():
            samples = diffusion.sample(batch_size=16)
            # samples in [-1,1], convert to [0,1]
            grid = (samples + 1.0) / 2.0
            utils.save_image(grid, os.path.join(save_path, f'samples_epoch_{epoch}.png'), nrow=4)

    print('Training finished')
