import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict

from models.layers import Attention, SwiGLU, rms_norm
from models.vjepa.layers import RotaryEmbedding3D, apply_rotary_pos_emb_3d

class PatchEmbed3D(nn.Module):
    """
    Video to 3D Patch Embedding.
    Input: (bs, T, C, H, W)
    Output: (bs, t_p * h_p * w_p, D)
    """
    def __init__(self, patch_size=(2, 16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (bs, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4) # (bs, C, T, H, W)
        x = self.proj(x) # (bs, D, t_p, h_p, w_p)
        bs, d, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) # (bs, t*h*w, D)
        return x, (t, h, w)

class VisionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion, norm_eps=1e-5):
        super().__init__()
        self.attn = Attention(
            hidden_size=dim,
            head_dim=dim // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(hidden_size=dim, expansion=expansion)
        self.norm_eps = norm_eps

    def forward(self, x, cos_sin):
        # x: (bs, seq_len, dim)
        x = rms_norm(x + self.attn(cos_sin, x), self.norm_eps)
        x = rms_norm(x + self.mlp(x), self.norm_eps)
        return x

class VisionEncoder(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=(2, 16, 16), 
                 in_chans=3, 
                 embed_dim=1024, 
                 depth=12, 
                 num_heads=16, 
                 expansion=4.0,
                 max_t=16,
                 max_h=14, # 224/16
                 max_w=14):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)
        
        self.rope = RotaryEmbedding3D(
            dim=embed_dim // num_heads,
            max_t=max_t,
            max_h=max_h,
            max_w=max_w
        )

        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, expansion)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (bs, T, C, H, W)
        x, (t, h, w) = self.patch_embed(x)
        cos_sin = self.rope(t, h, w)

        for block in self.blocks:
            x = block(x, cos_sin)
        
        return self.norm(x)
