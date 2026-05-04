"""
Information Bottleneck for Perception.

The Information Bottleneck (IB) principle: compress the input X into a
representation Z that captures only the information relevant to predicting Y.

  min I(X; Z) - β * I(Z; Y)

Where:
  I(X; Z) = mutual information between input and representation (compression)
  I(Z; Y) = mutual information between representation and target (prediction)
  β = tradeoff parameter (higher = more prediction, lower = more compression)

This implements the core principle of human perception: see only what matters.

Based on:
  - "The Information Bottleneck Method" (Tishby, Pereira, Bialek, 1999)
  - "Deep Learning and the Information Bottleneck Principle" (Tishby, Zaslavsky, 2015)
  - "Variational Information Bottleneck" (Alemi et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class VariationalInformationBottleneck(nn.Module):
    """
    Variational Information Bottleneck (VIB) layer.

    Compresses input through a learned Gaussian bottleneck, then
    reconstructs the task-relevant information. The KL regularization
    forces the model to discard irrelevant information.

    Args:
        input_dim: dimensionality of input features.
        bottleneck_dim: dimensionality of the compressed representation.
        beta: IB tradeoff (higher = more compression, lower = more prediction).
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        beta: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta

        # Encoder: input -> Gaussian parameters (mean, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, bottleneck_dim * 2),  # mean + log_var
        )

        # Decoder: bottleneck -> output
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )

        # Initialize near-identity
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def _reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: sample z = mean + std * epsilon
        where epsilon ~ N(0, 1).

        This allows gradients to flow through the sampling.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # During inference, just use the mean
            return mean

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (bs, ..., input_dim) input features.
            target: (bs, ..., input_dim) optional target for supervised IB.

        Returns:
            Dictionary with:
              - 'output': (bs, ..., input_dim) reconstructed features
              - 'z': (bs, ..., bottleneck_dim) compressed representation
              - 'kl_loss': scalar KL divergence loss
              - 'mean': (bs, ..., bottleneck_dim) posterior mean
              - 'log_var': (bs, ..., bottleneck_dim) posterior log variance
        """
        # Encode to Gaussian parameters
        params = self.encoder(x)
        mean, log_var = params.chunk(2, dim=-1)

        # Clamp log_var for stability
        log_var = log_var.clamp(-10, 10)

        # Sample from bottleneck
        z = self._reparameterize(mean, log_var)

        # Decode
        output = self.decoder(z)

        # KL divergence: KL(q(z|x) || p(z))
        # where p(z) = N(0, I) and q(z|x) = N(mean, exp(log_var))
        kl_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        kl_loss = kl_loss.sum(dim=-1).mean()  # sum over bottleneck dim, mean over batch

        return {
            'output': output,
            'z': z,
            'kl_loss': self.beta * kl_loss,
            'mean': mean,
            'log_var': log_var,
        }


class InformationBottleneckAttention(nn.Module):
    """
    Attention mechanism with Information Bottleneck regularization.

    Instead of standard softmax attention, this compresses the attention
    distribution through an information bottleneck, forcing the model
    to attend only to the most relevant tokens.

    Args:
        dim: model dimensionality.
        num_heads: number of attention heads.
        bottleneck_ratio: compression ratio for the attention bottleneck.
        beta: IB tradeoff parameter.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bottleneck_ratio: float = 0.5,
        beta: float = 1e-3,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Standard QKV projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Information bottleneck on attention scores
        bottleneck_dim = max(1, int(self.head_dim * bottleneck_ratio))
        self.ib = VariationalInformationBottleneck(
            input_dim=self.head_dim,
            bottleneck_dim=bottleneck_dim,
            beta=beta,
        )

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (bs, seq_len, dim) input.
            cos_sin: optional RoPE embeddings.
            target: optional target for supervised IB.

        Returns:
            output: (bs, seq_len, dim) attention output.
            info: dictionary with IB diagnostics.
        """
        bs, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale / self.temperature.abs()

        # Apply information bottleneck to attention
        # Reshape for IB: (bs * num_heads * seq_len, seq_len)
        attn_flat = attn_scores.reshape(-1, seq_len)

        # Compress attention through bottleneck
        ib_out = self.ib(attn_flat, target=attn_flat if target is not None else None)

        # Use compressed representation to compute attention weights
        attn_weights = F.softmax(ib_out['output'].reshape(bs, self.num_heads, seq_len, seq_len), dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(bs, seq_len, self.dim)
        out = self.o_proj(out)

        info = {
            'kl_loss': ib_out['kl_loss'],
            'bottleneck_mean': ib_out['mean'],
            'bottleneck_log_var': ib_out['log_var'],
        }

        return out, info
