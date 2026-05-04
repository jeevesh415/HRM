"""
Conditional Flow Matching for simulation-free continuous-time generative modeling.

Flow Matching (Lipman et al., 2023; Esser et al., 2024) is a simulation-free
alternative to diffusion models and Neural ODEs. Instead of learning a score
function or solving ODEs during training, it directly regresses the conditional
vector field between a noise source and target data via straight-line
interpolation paths.

This module can serve as an alternative or complement to the Neural ODE-based
physics engine (ContinuousTimeHRM), offering:
  - Simulation-free training (no ODE solver in the forward pass)
  - Nearly straight transport paths (Rectified Flow / reflow)
  - 1-2 step generation after reflow
  - Stable training with simple MSE loss

Based on:
  - "Flow Matching for Generative Modeling" (Lipman et al., 2023)
  - "Scaling Rectified Flow Transformers" (Esser et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (bs, 1) continuous time in [0, 1].

        Returns:
            (bs, dim) sinusoidal embedding.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * torch.log(torch.tensor(10000.0))
            / half
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityField(nn.Module):
    """
    Neural network that predicts the velocity field v(x_t, t, condition).

    Uses sinusoidal time embedding, FiLM conditioning, and a residual MLP
    for stable training.
    """

    def __init__(self, dim: int, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(dim // 4)
        self.time_proj = nn.Sequential(
            nn.Linear(dim // 4, hidden_dim),
            nn.SiLU(),
        )

        self.cond_proj = nn.Linear(condition_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

        # FiLM layers for time and condition injection
        self.film_time = nn.Linear(hidden_dim, dim)
        self.film_cond = nn.Linear(hidden_dim, dim)

        # Residual scaling (zero-init for stable start)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (bs, dim) noisy state at time t.
            t: (bs, 1) continuous time in [0, 1].
            condition: (bs, condition_dim) conditioning signal.

        Returns:
            (bs, dim) predicted velocity.
        """
        t_emb = self.time_proj(self.time_embed(t))  # (bs, hidden)
        c_emb = self.cond_proj(condition)  # (bs, hidden)

        h = self.net[0](x_t)  # first linear
        h = self.net[1](h)  # first silu

        # FiLM modulation from time and condition
        t_scale = self.film_time(t_emb).unsqueeze(1) if t_emb.ndim < h.ndim else self.film_time(t_emb)
        c_scale = self.film_cond(c_emb).unsqueeze(1) if c_emb.ndim < h.ndim else self.film_cond(c_emb)

        # Apply FiLM: scale and shift
        h = h * (1 + t_scale) + c_scale

        # Remaining layers
        h = self.net[2](h)
        h = self.net[3](h)
        h = self.net[4](h)
        h = self.net[5](h)
        return self.net[6](h)


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching for simulation-free continuous-time dynamics.

    Learns a velocity field v(x_t, t, c) that transports samples from a
    source distribution (Gaussian noise) to a target distribution. Training
    uses straight-line interpolation paths:

        x_t = (1 - t) * x_0 + t * x_1

    with target velocity u_t = x_1 - x_0 (constant along the path).

    After training, generation integrates the learned velocity field:
        dx/dt = v(x, t, c)

    For nearly straight paths (reflow / Rectified Flow), 1-2 step
    generation is possible via sample_rectified().

    Args:
        dim: dimensionality of the data space.
        hidden_dim: hidden dimension for the velocity network.
        condition_dim: dimensionality of the conditioning signal.
        sigma_min: minimum noise level (unused but reserved for OT-CFM).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        condition_dim: Optional[int] = None,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.dim = dim
        self.sigma_min = sigma_min
        hidden_dim = hidden_dim or dim * 2
        condition_dim = condition_dim or dim

        self.velocity_net = VelocityField(dim, hidden_dim, condition_dim)

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching training loss.

        Args:
            x_0: source noise, shape (bs, dim).
            x_1: target data, shape (bs, dim).
            condition: optional conditioning, shape (bs, condition_dim).

        Returns:
            Dictionary with 'loss', 'predicted_velocity', 'target_velocity'.
        """
        bs = x_0.shape[0]
        device = x_0.device

        # Sample random time t ~ U(0, 1)
        t = torch.rand(bs, 1, device=device)

        # Straight-line interpolation: x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * x_1

        # Target velocity is constant along the straight path
        target_velocity = x_1 - x_0

        # Default conditioning to zeros
        if condition is None:
            condition = torch.zeros(bs, self.dim, device=device, dtype=x_0.dtype)

        # Predict velocity field
        predicted_velocity = self.velocity_net(x_t, t, condition)

        # MSE loss on the velocity field
        loss = F.mse_loss(predicted_velocity, target_velocity)

        return {
            "loss": loss,
            "predicted_velocity": predicted_velocity,
            "target_velocity": target_velocity,
        }

    @torch.no_grad()
    def sample(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        steps: int = 50,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Generate samples by integrating the learned velocity field.

        Args:
            x_0: initial noise, shape (bs, dim).
            condition: optional conditioning, shape (bs, condition_dim).
            steps: number of integration steps.
            method: 'euler' or 'midpoint' (Heun's method).

        Returns:
            Generated samples, shape (bs, dim).
        """
        if condition is None:
            condition = torch.zeros(x_0.shape[0], self.dim, device=x_0.device, dtype=x_0.dtype)

        dt = 1.0 / steps
        x = x_0

        for i in range(steps):
            t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)

            v = self.velocity_net(x, t, condition)

            if method == "euler":
                x = x + v * dt
            elif method == "midpoint":
                # Heun's method / midpoint
                x_mid = x + v * dt / 2
                t_mid = torch.full(
                    (x.shape[0], 1), (i + 0.5) * dt, device=x.device, dtype=x.dtype
                )
                v_mid = self.velocity_net(x_mid, t_mid, condition)
                x = x + v_mid * dt
            else:
                raise ValueError(f"Unknown integration method: {method}")

        return x

    @torch.no_grad()
    def sample_rectified(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Rectified Flow sampling with near-straight paths.

        After reflow training, the transport paths become nearly straight,
        enabling 1-step generation. This method evaluates the velocity at
        t=0 and takes a single step.

        Args:
            x_0: initial noise, shape (bs, dim).
            condition: optional conditioning, shape (bs, condition_dim).

        Returns:
            Generated samples in one step, shape (bs, dim).
        """
        if condition is None:
            condition = torch.zeros(x_0.shape[0], self.dim, device=x_0.device, dtype=x_0.dtype)

        t = torch.zeros(x_0.shape[0], 1, device=x_0.device, dtype=x_0.dtype)
        v = self.velocity_net(x_0, t, condition)
        return x_0 + v
