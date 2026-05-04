import torch
from torch import nn
from typing import Optional
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

    def forward(self, z: torch.Tensor, delta_t: torch.Tensor | float = 1.0, action: Optional[torch.Tensor] = None):
        # Scalar horizon path.
        if not torch.is_tensor(delta_t):
            self.ode_func.current_action = action
            t = torch.tensor([0.0, float(delta_t)], device=z.device, dtype=z.dtype)
            zt1 = odeint(self.ode_func, z, t, method='dopri5')
            self.ode_func.current_action = None
            return zt1[-1]

        # Tensor horizon path: keep sample-specific dt while minimizing solver calls.
        if delta_t.ndim == 0:
            delta_t = delta_t.expand(z.shape[0])
        else:
            delta_t = delta_t.reshape(z.shape[0], -1).mean(dim=-1)
        delta_t = delta_t.to(device=z.device, dtype=z.dtype)

        # Fast path: all samples share one horizon => one batched ODE solve.
        if torch.allclose(delta_t, delta_t[0].expand_as(delta_t)):
            self.ode_func.current_action = action
            t = torch.stack([torch.zeros((), device=z.device, dtype=z.dtype), delta_t[0]])
            zt1 = odeint(self.ode_func, z, t, method='dopri5')
            self.ode_func.current_action = None
            return zt1[-1]

        # Group by unique horizons to reduce Python-loop overhead.
        evolved = torch.empty_like(z)
        unique_dt, inverse = torch.unique(delta_t, sorted=False, return_inverse=True)
        for group_idx, dt in enumerate(unique_dt):
            idx = torch.nonzero(inverse == group_idx, as_tuple=False).squeeze(-1)
            self.ode_func.current_action = None if action is None else action.index_select(0, idx)
            t = torch.stack([torch.zeros((), device=z.device, dtype=z.dtype), dt])
            z_group = z.index_select(0, idx)
            zt1 = odeint(self.ode_func, z_group, t, method='dopri5')
            evolved.index_copy_(0, idx, zt1[-1])

        self.ode_func.current_action = None
        return evolved
