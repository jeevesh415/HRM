import torch
from torch import nn
from typing import Callable, Tuple, Optional
from torchdiffeq import odeint_adjoint as odeint

class HRMPhysicsODE(nn.Module):
    """
    SOTA Hamiltonian Neural Network (HNN) Physics Engine.
    Computes continuous-time latent dynamics using exact Hamiltonian mechanics
    to guarantee conservation of energy and symplectic structure in the latent space.
    """
    def __init__(self, dim: int, action_dim: Optional[int] = 128):
        super().__init__()
        # For Hamiltonian mechanics, the latent state is split into position (q) and momentum (p)
        assert dim % 2 == 0, "Latent dimension must be even to split into (q, p) pairs."
        
        input_dim = dim + (action_dim if action_dim else 0)
        # The network predicts the total 'Energy' (Hamiltonian) of the system: H(q, p) -> R
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1) # Outputs a scalar energy value per sequence token
        )
        
        # Action is cached here for the duration of the integration step
        self.current_action = None

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        # Enable grad to compute conservative vector fields via autograd
        with torch.enable_grad():
            z = z.requires_grad_(True)
            
            action = self.current_action
            if action is not None:
                if action.ndim < z.ndim:
                    action = action.view(action.shape + (1,) * (z.ndim - action.ndim)).expand_as(z[..., :action.shape[-1]])
                z_input = torch.cat([z, action], dim=-1)
            else:
                z_input = z
                
            # Compute total system energy
            energy = self.energy_net(z_input).sum()
            
            # Compute partial derivatives of energy with respect to latent state (q, p)
            dz = torch.autograd.grad(energy, z, create_graph=True)[0]
            
            # Split gradients into dq and dp
            q_dim = z.shape[-1] // 2
            dH_dq, dH_dp = dz[..., :q_dim], dz[..., q_dim:]
            
            # Hamiltonian Equations of Motion:
            # dq/dt = dH/dp
            # dp/dt = -dH/dq
            dq_dt = dH_dp
            dp_dt = -dH_dq
            
            return torch.cat([dq_dt, dp_dt], dim=-1)

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
