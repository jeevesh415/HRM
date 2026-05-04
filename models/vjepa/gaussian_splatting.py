"""
3D Gaussian Splatting in Latent Space.

Replaces the NeRF-based LatentRayMarcher with explicit Gaussian primitives
for faster and higher-quality latent rendering. Each Gaussian is defined by
a position, anisotropic scale, rotation (quaternion), opacity, and a latent
feature vector. Rendering splats features onto query points using weighted
sums of Gaussian kernels.

Based on:
  - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    (Kerbl et al., 2023)
  - "Hybrid Latents" (2026)

Advantages over NeRF-based ray marching:
  - Explicit primitives (no volumetric integration)
  - Faster rendering (closed-form splatting)
  - Better gradient flow (direct parameter optimization)
  - Anisotropic kernels capture directional structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LatentGaussianSplatting(nn.Module):
    """
    3D Gaussian Splatting renderer operating in latent space.

    Predicts a set of 3D Gaussians from the global context and uses them
    to render latent features at arbitrary query positions via splatting.

    Args:
        dim: latent feature dimensionality.
        num_gaussians: number of Gaussian primitives to use.
    """

    def __init__(self, dim: int, num_gaussians: int = 256):
        super().__init__()
        self.dim = dim
        self.num_gaussians = num_gaussians

        # Total parameter count per Gaussian:
        #   position(3) + scale(3) + rotation(4) + opacity(1) + feature(dim)
        self.params_per_gaussian = 3 + 3 + 4 + 1 + dim

        # Predict Gaussian parameters from global context
        self.gaussian_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, num_gaussians * self.params_per_gaussian),
        )

        # Project latent points to 3D positions
        self.latent_to_3d = nn.Linear(dim, 3)

        # Feature rendering head
        self.feature_renderer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Initialize near-zero for stable early training
        nn.init.zeros_(self.gaussian_encoder[-1].weight)
        nn.init.zeros_(self.gaussian_encoder[-1].bias)

    def _parse_gaussians(self, params: torch.Tensor) -> dict:
        """
        Parse a flat parameter vector into structured Gaussian components.

        Args:
            params: (bs, num_gaussians * params_per_gaussian)

        Returns:
            Dictionary with keys: means, scales, rotations, opacities, features.
        """
        bs = params.shape[0]
        params = params.reshape(bs, self.num_gaussians, self.params_per_gaussian)

        idx = 0

        # Position (unconstrained)
        means = params[..., idx : idx + 3]
        idx += 3

        # Scale (softplus to ensure positivity, clamped for stability)
        scales = F.softplus(params[..., idx : idx + 3]).clamp(min=1e-4, max=10.0)
        idx += 3

        # Rotation as unit quaternion
        rotations = F.normalize(params[..., idx : idx + 4], dim=-1)
        idx += 4

        # Opacity (sigmoid to [0, 1])
        opacities = torch.sigmoid(params[..., idx : idx + 1])
        idx += 1

        # Per-Gaussian latent features
        features = params[..., idx : idx + self.dim]

        return {
            "means": means,          # (bs, N, 3)
            "scales": scales,        # (bs, N, 3)
            "rotations": rotations,  # (bs, N, 4)
            "opacities": opacities,  # (bs, N, 1)
            "features": features,    # (bs, N, dim)
        }

    @staticmethod
    def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """
        Convert unit quaternions to 3x3 rotation matrices.

        Args:
            q: (..., 4) unit quaternions (w, x, y, z).

        Returns:
            (..., 3, 3) rotation matrices.
        """
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)
        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)
        R = torch.stack(
            [torch.stack([r00, r01, r02], dim=-1),
             torch.stack([r10, r11, r12], dim=-1),
             torch.stack([r20, r21, r22], dim=-1)],
            dim=-2,
        )
        return R

    def forward(
        self,
        latents: torch.Tensor,
        ray_dirs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render latent features using 3D Gaussian splatting.

        Args:
            latents: (bs, num_points, dim) input latent points.
            ray_dirs: (bs, num_points, 3) ray directions (unused, for API
                      compatibility with LatentRayMarcher).

        Returns:
            (bs, num_points, dim) rendered latent features.
        """
        bs, n, d = latents.shape

        # Global context via mean pooling
        global_context = latents.mean(dim=1)  # (bs, dim)

        # Predict Gaussian parameters
        gaussian_params = self.gaussian_encoder(global_context)  # (bs, N*param_size)
        gaussians = self._parse_gaussians(gaussian_params)

        # Project query latents to 3D positions
        points_3d = self.latent_to_3d(latents)  # (bs, n, 3)

        # Expand for broadcasting
        # Gaussians: (bs, N, 1, 3) ; Points: (bs, 1, n, 3)
        means = gaussians["means"].unsqueeze(2)       # (bs, N, 1, 3)
        scales = gaussians["scales"].unsqueeze(2)     # (bs, N, 1, 3)
        points = points_3d.unsqueeze(1)               # (bs, 1, n, 3)

        # Compute Mahalanobis distance with diagonal covariance
        # (ignoring rotation for efficiency; rotation can be applied
        # via the covariance matrix if needed for higher fidelity)
        diff = (points - means) / (scales + 1e-6)  # (bs, N, n, 3)
        mahal_dist = (diff ** 2).sum(dim=-1)        # (bs, N, n)

        # Gaussian kernel: exp(-0.5 * d^2)
        weights = torch.exp(-0.5 * mahal_dist)      # (bs, N, n)

        # Weight by opacity
        opacities = gaussians["opacities"].squeeze(-1).unsqueeze(-1)  # (bs, N, 1)
        weights = weights * opacities

        # Normalize across Gaussians for each query point
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        # Splat features: weighted sum of Gaussian features
        gaussian_features = gaussians["features"].unsqueeze(2)  # (bs, N, 1, dim)
        rendered = (weights.unsqueeze(-1) * gaussian_features).sum(dim=1)  # (bs, n, dim)

        # Final rendering refinement
        output = self.feature_renderer(rendered)

        return output
