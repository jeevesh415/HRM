import torch
from torch import nn
from typing import Callable, Tuple

class RK4Solver(nn.Module):
    """
    Runge-Kutta 4th Order Solver for Neural ODEs.
    Allows the model to learn continuous-time physical dynamics.
    """
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, z0: torch.Tensor, t0: float, t1: float, steps: int = 4) -> torch.Tensor:
        h = (t1 - t0) / steps
        z = z0
        t = t0
        
        for _ in range(steps):
            k1 = self.func(t, z)
            k2 = self.func(t + h/2, z + h/2 * k1)
            k3 = self.func(t + h/2, z + h/2 * k2)
            k4 = self.func(t + h, z + h * k3)
            
            z = z + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t = t + h
            
        return z

class HRMPhysicsODE(nn.Module):
    """
    The 'Derivative' function for our Neural ODE.
    It computes the change in the latent state (dz/dt) based on current state and input.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        # Time-conditioned latent update
        # We can also inject 't' into the network if needed
        return self.norm(self.net(z))

class ContinuousTimeHRM(nn.Module):
    """
    Upgraded HRM cycle that uses Neural ODEs for continuous-time reasoning.
    Instead of discrete steps, it 'evolves' the latent state over a time interval.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.ode_func = HRMPhysicsODE(dim)
        self.solver = RK4Solver(self.ode_func)

    def forward(self, z: torch.Tensor, delta_t: float = 1.0):
        # Evolve the hidden state z forward in time by delta_t
        return self.solver(z, t0=0.0, t1=delta_t)
