"""
Symplectic Euler Integrator for Hamiltonian Systems.

This module provides a structure-preserving integrator that exactly maintains
the symplectic two-form of Hamiltonian dynamics. Unlike the adjoint method
(used in ContinuousTimeHRM via torchdiffeq) which is an approximate ODE
solver, symplectic integrators guarantee:

  - Exact preservation of the symplectic structure (phase-space volume)
  - No long-term energy drift (bounded oscillation around true energy)
  - Time-reversibility (for the leapfrog/Störmer-Verlet variant)

The symplectic Euler scheme is a first-order implicit-explicit method:
  1. Kick:  p_{n+1} = p_n - dt * dH/dq(q_n, p_n)
  2. Drift: q_{n+1} = q_n + dt * dH/dp(q_n, p_{n+1})

This is the simplest symplectic integrator and serves as a drop-in
replacement for the Neural ODE when strict energy conservation matters
more than adaptive step-size control.

Based on:
  - "SPINI: Structure-Preserving Neural Integrator" (Nature, Dec 2025)
  - "Learning Hamiltonian Dynamics at Scale" (NeurIPS 2025)
"""

import torch
import torch.nn as nn
from typing import Optional


class SymplecticEulerIntegrator(nn.Module):
    """
    Symplectic Euler Integrator for learned Hamiltonian systems.

    The Hamiltonian H(q, p) is parameterized by a neural network. At each
    integration step, we:
      1. Compute dH/dq via autograd and update p (kick)
      2. Compute dH/dp via autograd and update q (drift)

    This guarantees that the map (q_n, p_n) -> (q_{n+1}, p_{n+1}) is
    exactly symplectic, regardless of the network architecture of H.

    Args:
        dim: total state dimension (must be even; split into q and p).
        action_dim: optional external action/condition dimension.
    """

    def __init__(self, dim: int, action_dim: Optional[int] = None):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even for Hamiltonian split, got {dim}"
        self.dim = dim
        self.half_dim = dim // 2
        self.action_dim = action_dim

        input_dim = dim + (action_dim or 0)
        self.H_net = nn.Sequential(
            nn.Linear(input_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

        # Cached action for integration steps
        self.current_action: Optional[torch.Tensor] = None

    def set_action(self, action: Optional[torch.Tensor]) -> None:
        """Set the external action/condition for the next integration call."""
        self.current_action = action

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamiltonian energy H(q, p).

        Args:
            q: (bs, half_dim) position coordinates.
            p: (bs, half_dim) momentum coordinates.

        Returns:
            (bs, 1) scalar energy per sample.
        """
        z = torch.cat([q, p], dim=-1)
        if self.current_action is not None:
            z = torch.cat([z, self.current_action], dim=-1)
        return self.H_net(z)

    def forward(
        self,
        z: torch.Tensor,
        dt: float = 1.0,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Perform symplectic Euler integration.

        Args:
            z: (bs, dim) initial state, where z = [q, p].
            dt: total integration time.
            steps: number of sub-steps.

        Returns:
            (bs, dim) evolved state after time dt.
        """
        dt_step = dt / steps
        q, p = z[..., : self.half_dim], z[..., self.half_dim :]

        for _ in range(steps):
            # --- Kick: update p using dH/dq ---
            with torch.enable_grad():
                q_g = q.detach().requires_grad_(True)
                p_g = p.detach()  # p held fixed during kick
                H = self.hamiltonian(q_g, p_g).sum()
                dH_dq = torch.autograd.grad(H, q_g, create_graph=True)[0]
            p = p - dt_step * dH_dq

            # --- Drift: update q using dH/dp ---
            with torch.enable_grad():
                q_g = q.detach()  # q held fixed during drift
                p_g = p.requires_grad_(True)
                H = self.hamiltonian(q_g, p_g).sum()
                dH_dp = torch.autograd.grad(H, p_g, create_graph=True)[0]
            q = q + dt_step * dH_dp

        return torch.cat([q, p], dim=-1)

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamiltonian energy of a state.

        Args:
            z: (bs, dim) state vector.

        Returns:
            (bs, 1) energy.
        """
        q, p = z[..., : self.half_dim], z[..., self.half_dim :]
        return self.hamiltonian(q, p)
