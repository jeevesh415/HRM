"""
Multi-Modal Grounding for Human-Like World Understanding.

Humans don't just see — we hear, touch, and feel. This module provides
the architectural hooks for grounding visual representations in:
  - Audio: sound provides causal information (a crash, a splash)
  - Tactile/Proprioceptive: touch provides physical properties (soft, hot, heavy)

The grounding uses cross-modal attention: visual features attend to audio
features, and vice versa. This creates a unified representation where
seeing a ball bounce is linked to hearing the bounce and feeling the impact.

Based on:
  - "ImageBind" (Girdhar et al., 2023) — one embedding space for all modalities
  - "Perceiver" (Jaegle et al., 2021) — cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ModalityEncoder(nn.Module):
    """
    Encodes a single modality (audio, tactile) into the shared latent space.

    Uses a simple MLP + positional encoding to map raw modality features
    into the same dimensionality as the visual latent space.

    Args:
        input_dim: dimensionality of raw modality features.
        output_dim: dimensionality of the shared latent space.
        max_seq_len: maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Projection to shared space
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, output_dim) * 0.02)

        # Layer norm
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (bs, input_dim) or (bs, seq_len, input_dim) raw features.
            seq_len: optional sequence length for positional encoding.

        Returns:
            (bs, seq_len, output_dim) encoded features.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (bs, 1, input_dim)

        bs, sl, _ = x.shape

        # Project
        h = self.proj(x)

        # Add positional encoding
        if sl <= self.pos_enc.shape[0]:
            h = h + self.pos_enc[:sl].unsqueeze(0)

        return self.norm(h)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: one modality attends to another.

    Visual features query audio/tactile features to find relevant
    cross-modal associations. This is how "seeing a ball" connects
    to "hearing a bounce".

    Args:
        dim: shared latent dimensionality.
        num_heads: number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query from source modality, Key/Value from target modality
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Gating for residual connection
        self.gate = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # start with low gate value

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source: (bs, seq_s, dim) features that query.
            target: (bs, seq_t, dim) features being attended to.

        Returns:
            (bs, seq_s, dim) cross-modal attended features.
        """
        bs, seq_s, _ = source.shape
        seq_t = target.shape[1]

        # Q from source, K/V from target
        q = self.q_proj(source).view(bs, seq_s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(target).view(bs, seq_t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(target).view(bs, seq_t, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bs, seq_s, self.dim)
        out = self.o_proj(out)

        # Gated residual
        gate = torch.sigmoid(self.gate(source))
        return source + gate * out


class MultiModalGrounding(nn.Module):
    """
    Multi-modal grounding module that unifies visual, audio, and tactile
    representations into a single coherent world model.

    Architecture:
      1. Each modality is encoded into the shared latent space
      2. Cross-modal attention links modalities bidirectionally
      3. A fusion layer creates a unified multi-modal representation

    This enables the model to:
      - Infer sounds from visual cues (seeing a crash → expecting a bang)
      - Infer physical properties from appearance (seeing water → expecting wetness)
      - Use audio to disambiguate visual scenes (hearing music → seeing a concert)

    Args:
        dim: shared latent dimensionality.
        num_heads: number of attention heads.
        audio_input_dim: dimensionality of raw audio features.
        tactile_input_dim: dimensionality of raw tactile features.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        audio_input_dim: int = 128,
        tactile_input_dim: int = 64,
    ):
        super().__init__()
        self.dim = dim

        # Modality encoders
        self.audio_encoder = ModalityEncoder(audio_input_dim, dim)
        self.tactile_encoder = ModalityEncoder(tactile_input_dim, dim)

        # Cross-modal attention (bidirectional)
        # Visual → Audio (what does this scene sound like?)
        self.visual_to_audio = CrossModalAttention(dim, num_heads)
        # Visual → Tactile (what does this scene feel like?)
        self.visual_to_tactile = CrossModalAttention(dim, num_heads)
        # Audio → Visual (what does this sound look like?)
        self.audio_to_visual = CrossModalAttention(dim, num_heads)

        # Fusion layer: combines all modalities
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        # Modality presence indicators (handle missing modalities)
        self.audio_present = nn.Parameter(torch.tensor(0.0))
        self.tactile_present = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        tactile_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (bs, seq_v, dim) visual latent features.
            audio_features: (bs, audio_dim) or (bs, seq_a, audio_dim) audio.
            tactile_features: (bs, tactile_dim) or (bs, seq_t, tactile_dim) tactile.

        Returns:
            (bs, seq_v, dim) grounded visual features with multi-modal context.
        """
        bs, seq_v, dim = visual_features.shape

        # Start with visual features
        grounded = visual_features.clone()

        # Ground in audio if available
        if audio_features is not None:
            audio_encoded = self.audio_encoder(audio_features)
            # Visual attends to audio
            visual_audio = self.visual_to_audio(visual_features, audio_encoded)
            # Audio attends to visual
            audio_visual = self.audio_to_visual(audio_encoded, visual_features)

            # Add audio grounding to visual
            gate = torch.sigmoid(self.audio_present)
            grounded = grounded + gate * visual_audio

        # Ground in tactile if available
        if tactile_features is not None:
            tactile_encoded = self.tactile_encoder(tactile_features)
            # Visual attends to tactile
            visual_tactile = self.visual_to_tactile(visual_features, tactile_encoded)

            gate = torch.sigmoid(self.tactile_present)
            grounded = grounded + gate * visual_tactile

        # If no extra modalities, just return visual features
        if audio_features is None and tactile_features is None:
            return visual_features

        return grounded
