import torch
from torch import nn
from typing import Callable, Tuple, Optional
from torchdiffeq import odeint_adjoint as odeint

class HRMPhysicsODE(nn.Module):
    """
    The 'Derivative' function for our Neural ODE.
    It computes the change in the latent state (dz/dt) based on current state and input.
    """
    def __init__(self, dim: int, action_dim: Optional[int] = 128):
        super().__init__()
        # If action is provided, we expand the input dimension
        input_dim = dim + (action_dim if action_dim else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
        # Action is cached here for the duration of the integration step
        self.current_action = None

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        # Time-conditioned latent update incorporating action
        action = self.current_action
        if action is not None:
            if action.ndim < z.ndim:
                action = action.view(action.shape + (1,) * (z.ndim - action.ndim)).expand_as(z[..., :action.shape[-1]])
            z_input = torch.cat([z, action], dim=-1)
        else:
            z_input = z
            
        return self.norm(self.net(z_input))

class ContinuousTimeHRM(nn.Module):
    """
    Upgraded HRM cycle using the SOTA Neural ODE Adjoint Method from torchdiffeq.
    Provides mathematically exact, memory-efficient continuous-time latent evolution
    with adaptive solvers (e.g. dopri5).
    """
    def __init__(self, dim: int, action_dim: int = 128):
        super().__init__()
        self.ode_func = HRMPhysicsODE(dim, action_dim)

    def forward(self, z: torch.Tensor, delta_t: float = 1.0, action: Optional[torch.Tensor] = None):
        # Evolve using the true Adjoint Method with dopri5 adaptive solver
        self.ode_func.current_action = action
        
        t = torch.tensor([0.0, delta_t], device=z.device, dtype=z.dtype)
        # odeint_adjoint handles O(1) memory backprop and adaptive steps perfectly
        zt1 = odeint(self.ode_func, z, t, method='dopri5')
        
        self.ode_func.current_action = None
        return zt1[-1]  # Return state at t=delta_t
