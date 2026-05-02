import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class RotaryEmbedding3D(nn.Module):
    """
    3D Rotary Positional Embedding (3D-RoPE) for spatio-temporal data.
    Splits the head dimension into three parts for Time (T), Height (H), and Width (W).
    """
    def __init__(self, dim: int, max_t: int, max_h: int, max_w: int, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.base = base

        # Split dim into 3 parts. 
        # We try to distribute them somewhat evenly, but give more to H/W if possible.
        self.dim_t = dim // 4
        self.dim_h = (dim - self.dim_t) // 2
        self.dim_w = dim - self.dim_t - self.dim_h

        self._build_cache(device)

    def _get_freqs(self, length: int, dim: int, device):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(length, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    def _build_cache(self, device):
        # We cache the 1D freqs for each axis.
        # When forwarding, we will broadcast and combine them based on the input grid.
        self.register_buffer("freqs_t", self._get_freqs(self.max_t, self.dim_t, device), persistent=False)
        self.register_buffer("freqs_h", self._get_freqs(self.max_h, self.dim_h, device), persistent=False)
        self.register_buffer("freqs_w", self._get_freqs(self.max_w, self.dim_w, device), persistent=False)

    def forward(self, t: int, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # t, h, w are the grid dimensions of the input patches
        # Extract slices from cache
        f_t = self.freqs_t[:t]  # [t, dim_t]
        f_h = self.freqs_h[:h]  # [h, dim_h]
        f_w = self.freqs_w[:w]  # [w, dim_w]

        # Expand and concatenate to create a [t, h, w, dim] grid
        # f_t: [t, 1, 1, dim_t]
        # f_h: [1, h, 1, dim_h]
        # f_w: [1, 1, w, dim_w]
        f_t = f_t.view(t, 1, 1, self.dim_t).expand(t, h, w, self.dim_t)
        f_h = f_h.view(1, h, 1, self.dim_h).expand(t, h, w, self.dim_h)
        f_w = f_w.view(1, 1, w, self.dim_w).expand(t, h, w, self.dim_w)

        # Concatenate along the last dimension
        grid_freqs = torch.cat([f_t, f_h, f_w], dim=-1) # [t, h, w, dim]
        
        # Flatten to [t*h*w, dim] for standard RoPE application
        grid_freqs = grid_freqs.reshape(-1, self.dim)

        return grid_freqs.cos(), grid_freqs.sin()

def apply_rotary_pos_emb_3d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] (flattened t*h*w)
    # This is identical to 1D application, just the cos/sin are constructed differently
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
