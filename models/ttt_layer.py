"""
Test-Time Training (TTT) Layer.

Replaces standard self-attention with a self-supervised inner loop that
adapts the model's hidden state during inference. The hidden state is
a linear model that learns to reconstruct each token from the context,
enabling real-time adaptation to new patterns.

This is the key mechanism for human-like reasoning: when faced with
something unfamiliar, the model "thinks harder" by running more
inner-loop steps, just as a human would focus more attention on
something confusing.

Based on:
  - "Learning to (Learn at Test Time)" (Sun et al., ICML 2025)
  - "Test-Time Training with KV Binding Is Secretly Linear Attention" (2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TTTLinear(nn.Module):
    """
    TTT-Linear layer: a sequence model where the hidden state is a
    linear model W that gets updated via gradient descent at each token.

    For each token x_t:
      1. Predict: x_hat_t = W @ x_t (reconstruction)
      2. Loss: L = ||x_hat_t - x_t||^2
      3. Update: W = W - lr * grad(L, W) (one gradient step)

    This makes the hidden state (W) an adaptive representation that
    captures the structure of the sequence as it processes it.

    Args:
        dim: model dimensionality.
        num_heads: number of parallel TTT heads.
        inner_lr: learning rate for the inner loop (learnable).
        num_inner_steps: number of inner-loop gradient steps per token.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        inner_lr: float = 0.1,
        num_inner_steps: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_inner_steps = num_inner_steps

        # QKV projections (same as attention)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Learnable inner learning rate
        self.inner_lr = nn.Parameter(torch.tensor(inner_lr))

        # The linear model weights (one per head)
        # W: (num_heads, head_dim, head_dim) — maps keys to values
        self.W = nn.Parameter(torch.zeros(num_heads, self.head_dim, self.head_dim))

        # Reconstruction loss weight
        self.recon_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (bs, seq_len, dim) input sequence.
            cos_sin: optional RoPE embeddings (unused for TTT, kept for API compat).

        Returns:
            (bs, seq_len, dim) output sequence.
        """
        bs, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (bs, num_heads, seq_len, head_dim)

        # Initialize the linear model W for this sequence
        W = self.W.unsqueeze(0).expand(bs, -1, -1, -1).clone()  # (bs, num_heads, head_dim, head_dim)

        # Process each token sequentially (TTT is inherently sequential)
        outputs = []
        lr = self.inner_lr.abs()  # ensure positive

        for t in range(seq_len):
            k_t = k[:, :, t, :]  # (bs, num_heads, head_dim)
            v_t = v[:, :, t, :]  # (bs, num_heads, head_dim)

            # Predict value from key using current W
            v_hat = torch.bmm(
                W.reshape(bs * self.num_heads, self.head_dim, self.head_dim),
                k_t.reshape(bs * self.num_heads, self.head_dim, 1)
            ).reshape(bs, self.num_heads, self.head_dim)

            # Inner loop: update W to better reconstruct v_t from k_t
            for _ in range(self.num_inner_steps):
                # Reconstruction loss gradient
                error = v_hat - v_t  # (bs, num_heads, head_dim)

                # dL/dW = error * k_t^T (outer product)
                grad_W = torch.bmm(
                    error.reshape(bs * self.num_heads, self.head_dim, 1),
                    k_t.reshape(bs * self.num_heads, 1, self.head_dim)
                ).reshape(bs, self.num_heads, self.head_dim, self.head_dim)

                # Gradient descent step
                W = W - lr * grad_W

                # Recompute prediction after update
                v_hat = torch.bmm(
                    W.reshape(bs * self.num_heads, self.head_dim, self.head_dim),
                    k_t.reshape(bs * self.num_heads, self.head_dim, 1)
                ).reshape(bs, self.num_heads, self.head_dim)

            # Query-based output: use q_t to read from W
            # Output = W^T @ q_t (the linear model transforms queries)
            out_t = torch.bmm(
                W.transpose(-1, -2).reshape(bs * self.num_heads, self.head_dim, self.head_dim),
                q[:, :, t, :].reshape(bs * self.num_heads, self.head_dim, 1)
            ).reshape(bs, self.num_heads, self.head_dim)

            outputs.append(out_t)

        # Stack outputs
        output = torch.stack(outputs, dim=2)  # (bs, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2).reshape(bs, seq_len, self.dim)

        return self.o_proj(output)


class TTTLinearWithAttention(nn.Module):
    """
    Hybrid layer combining TTT-Linear with standard attention.

    Uses a learnable gate to blend TTT (for adaptive reasoning) with
    attention (for precise token-to-token interaction). This captures
    both the adaptive nature of TTT and the global context of attention.

    Args:
        dim: model dimensionality.
        num_heads: number of heads.
        inner_lr: TTT inner learning rate.
        num_inner_steps: TTT inner steps per token.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        inner_lr: float = 0.1,
        num_inner_steps: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # TTT branch
        self.ttt = TTTLinear(dim, num_heads, inner_lr, num_inner_steps)

        # Attention branch
        from models.layers import Attention
        self.attn = Attention(
            hidden_size=dim,
            head_dim=dim // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )

        # Gate to blend TTT and attention
        self.gate = nn.Linear(dim, 2, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # TTT branch (adaptive reasoning)
        ttt_out = self.ttt(x, cos_sin)

        # Attention branch (global context)
        attn_out = self.attn(cos_sin=cos_sin, hidden_states=x)

        # Gated blend
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        return gate_weights[..., 0:1] * ttt_out + gate_weights[..., 1:2] * attn_out
