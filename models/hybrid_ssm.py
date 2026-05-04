"""
Hybrid SSM-Attention Architecture for Long-Range Reasoning.

Implements a Selective State Space Model (Mamba-style) combined with
standard attention for precise local
reasoning. A learnable gate blends the two branches per-token.

Based on:
  - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    (Gu & Dao, 2023)
  - "Transformers are SSMs" / Mamba-2 (Dao & Gu, 2024)
  - "Titans + MIRAS" (Google, 2025)

The SSM branch handles infinite-context long-range dependencies while the
attention branch provides precise token-to-token interactions. The gated
fusion lets the model learn which mechanism is better for each token.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba-style) block.

    Processes sequences with selective scan. Current implementation is
    sequential for correctness; parallel scan (O(n)) can be implemented
    via Blelloch prefix sum.

    Args:
        dim: input/output dimensionality.
        d_state: SSM state dimension (default: 16).
        d_conv: local convolution kernel size (default: 4).
        expand: inner dimension expansion factor (default: 2).
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv

        d_inner = int(expand * dim)

        # Input projection: x -> (x_proj, gate)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)

        # Causal 1D convolution for local context
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM parameter projection: x -> (B, C, dt)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)

        # A parameter in log-space for numerical stability
        # Initialize as a diagonal-plus-HiPPO-style matrix
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection / dt bias)
        self.D = nn.Parameter(torch.ones(d_inner))

        # dt bias for stable initialization
        self.dt_bias = nn.Parameter(torch.zeros(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, seq_len, dim) input sequence.

        Returns:
            (bs, seq_len, dim) output sequence.
        """
        bs, seq_len, _ = x.shape

        # Input projection and split into main path and gate
        xz = self.in_proj(x)  # (bs, seq_len, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)

        # Causal 1D convolution
        x_conv = x_proj.transpose(1, 2)  # (bs, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[..., :seq_len].transpose(1, 2)  # (bs, seq_len, d_inner)
        x_conv = F.silu(x_conv)

        # Project to SSM parameters
        x_ssm = self.x_proj(x_conv)  # (bs, seq_len, 2*d_state + 1)
        B = x_ssm[..., : self.d_state]                          # (bs, seq_len, d_state)
        C = x_ssm[..., self.d_state : 2 * self.d_state]        # (bs, seq_len, d_state)
        dt = F.softplus(x_ssm[..., -1] + self.dt_bias)         # (bs, seq_len)

        # Continuous A matrix (negative for stability)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Selective scan (sequential for correctness)
        d_inner = x_conv.shape[-1]
        h = torch.zeros(bs, d_inner, self.d_state, device=x.device, dtype=x_conv.dtype)
        outputs = []

        for t in range(seq_len):
            # Discretize: A_bar = exp(A * dt)
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)  # (bs, 1, 1)
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)     # (bs, d_inner, d_state)
            B_t = B[:, t].unsqueeze(1)                    # (bs, 1, d_state)
            x_t = x_conv[:, t].unsqueeze(-1)              # (bs, d_inner, 1)

            # State update: h = A_bar * h + B * x
            h = A_bar * h + x_t * B_t

            # Output: y = C * h + D * x (skip connection)
            C_t = C[:, t].unsqueeze(1)  # (bs, 1, d_state)
            y_t = (h * C_t).sum(dim=-1) + self.D * x_conv[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (bs, seq_len, d_inner)

        # Gated output
        y = y * F.silu(z)

        return self.out_proj(y)


class HybridSSMAttentionBlock(nn.Module):
    """
    Hybrid block combining SSM for long-range and Attention for local reasoning.

    Each block runs both an SSM branch and an attention branch in parallel,
    then blends them with a learnable per-token gate. This allows the model
    to use O(n) SSM for global context and O(n^2) attention for precise
    token interactions where needed.

    Inspired by the "Titans" architecture (Google, 2025).

    Args:
        dim: model dimensionality.
        num_heads: number of attention heads.
        expansion: MLP expansion ratio.
        ssm_d_state: SSM state dimension.
        ssm_d_conv: SSM convolution kernel size.
        ssm_expand: SSM inner expansion factor.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion: float = 4.0,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
    ):
        super().__init__()
        self.dim = dim

        # SSM branch (long-range, O(n))
        self.ssm = SelectiveSSM(dim, d_state=ssm_d_state, d_conv=ssm_d_conv, expand=ssm_expand)
        self.ssm_norm = nn.LayerNorm(dim)

        # Attention branch (local, O(n^2))
        # We import here to avoid circular imports; the layers module defines Attention
        from models.layers import Attention
        self.attn = Attention(
            hidden_size=dim,
            head_dim=dim // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.attn_norm = nn.LayerNorm(dim)

        # Per-token gate: blend SSM and attention outputs
        self.gate = nn.Linear(dim, 2, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0)

        # MLP
        from models.layers import SwiGLU
        self.mlp = SwiGLU(dim, expansion)
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (bs, seq_len, dim) input.
            cos_sin: optional RoPE embeddings for attention.

        Returns:
            (bs, seq_len, dim) output.
        """
        # SSM branch
        ssm_out = self.ssm_norm(self.ssm(x))

        # Attention branch
        attn_out = self.attn_norm(self.attn(cos_sin=cos_sin, hidden_states=x))

        # Gated fusion
        gate_logits = self.gate(x)              # (bs, seq, 2)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (bs, seq, 2)

        fused = gate_weights[..., 0:1] * ssm_out + gate_weights[..., 1:2] * attn_out

        # Residual + MLP
        x = x + fused
        x = x + self.mlp(self.mlp_norm(x))

        return x
