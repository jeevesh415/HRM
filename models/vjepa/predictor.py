import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable
from torchdiffeq import odeint_adjoint as odeint

from models.layers import Attention, SwiGLU, RMSNorm
from models.vjepa.layers import apply_rotary_pos_emb_3d, LieGroupEquivariantLayer
from models.vjepa.memory import LatentMemoryPalace
from models.ttt_layer import TTTLinearWithAttention
from models.multimodal_grounding import MultiModalGrounding
from models.spectral_conv import SpectralGraphConv
from models.topological import TopologicalAwareness
from models.proper_equivariance import ProperSE3EquivariantLayer
from models.uncertainty import UncertaintyQuantification

class UnifiedVectorField(nn.Module):
    """
    The 'Fantastic Scientist' Brain (V-JEPA 2.1).
    A unified Hamiltonian Neural ODE vector field that integrates physics,
    geometry, hierarchical memory, and iterative reasoning.
    """
    def __init__(self, dim: int, action_dim: int, num_gaussians: int, h_cycles: int, l_cycles: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.q_dim = dim // 2
        
        # 1. Geometry: Physical Relativity (SO(3)/SE(3))
        self.equivariant_layer = LieGroupEquivariantLayer(dim)
        
        # 2. Memory: Latent Memory Palace (Hierarchical)
        self.memory_palace = LatentMemoryPalace(dim)
        
        # 3. Physics: The Energy Functional (Hamiltonian)
        self.energy_net = nn.Sequential(
            nn.Linear(dim + action_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1)
        )
        
        # 4. Reasoning: Hierarchical Predictive Coding
        self.H_attn = Attention(dim, dim // num_heads, num_heads, num_heads)
        self.L_attn = Attention(dim, dim // num_heads, num_heads, num_heads)
        
        # 5. Vision: Continuous Rendering Influence
        self.gaussian_params = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, num_gaussians * (3 + 3 + 4 + 1 + dim))
        )
        
        self.current_action = None

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        # z: (bs, num_tokens, dim)
        with torch.enable_grad():
            # Standard ODE practice: z should already be traceable
            # We avoid re-marking requires_grad inside the loop for efficiency
            
            # --- 1. Geometric Relativity ---
            z = self.equivariant_layer(z, torch.zeros(z.shape[0], 3, device=z.device))
            
            # --- 2. Memory Palace Navigation ---
            memory_recall = self.memory_palace(z)
            
            # --- 3. Hamiltonian Physics ---
            if self.current_action is not None:
                action = self.current_action
                if action.ndim == 2:
                    action = action.unsqueeze(1).expand(-1, z.shape[1], -1)
                z_input = torch.cat([z, action], dim=-1)
            else:
                z_input = z
                
            energy = self.energy_net(z_input).sum()
            dz_hamiltonian = torch.autograd.grad(energy, z, create_graph=True)[0]
            
            dH_dq, dH_dp = dz_hamiltonian[..., :self.q_dim], dz_hamiltonian[..., self.q_dim:]
            v_physics = torch.cat([dH_dp, -dH_dq], dim=-1)
            
            # --- 4. Hierarchical Refinement ---
            z_high = z + self.H_attn(None, z + memory_recall)
            z_low = z + self.L_attn(None, z_high)
            v_reasoning = (z_low - z)
            
            # --- 5. Visual Rendering Pressure ---
            v_vision = torch.tanh(self.gaussian_params(z).mean(dim=-1, keepdim=True)) * z
            
            return v_physics + 0.1 * v_reasoning + 0.05 * v_vision

class VJEPAPredictorInner(nn.Module):
    """
    ULTIMATE UNIFICATION: V-JEPA 2.1 'Fantastic Scientist' Predictor.
    Scale-down: 2.1B parameters (2048-dim, 20-depth).
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 expansion: float,
                 h_cycles: int,
                 l_cycles: int,
                 action_dim: int = 128,
                 num_gaussians: int = 256,
                 **kwargs):
        super().__init__()
        self.dim = dim

        # The Unified Brain
        self.unified_brain = UnifiedVectorField(dim, action_dim, num_gaussians, h_cycles, l_cycles, num_heads)

        # Advanced sensory frontiers
        self.ttt_layer = TTTLinearWithAttention(dim, num_heads)
        self.multimodal = MultiModalGrounding(dim, num_heads, 128, 64)
        
        # Stateful Norm
        self.norm = RMSNorm(dim)

    def forward(self,
                context_latents: torch.Tensor,
                target_queries: torch.Tensor,
                cos_sin: Tuple[torch.Tensor, torch.Tensor],
                delta_t: torch.Tensor,
                action: Optional[torch.Tensor] = None,
                **kwargs):
        
        # 1. Sensory Pre-processing
        context_latents = self.multimodal(context_latents, **kwargs)
        
        # 2. Unified Continuous Evolution
        self.unified_brain.current_action = action
        # Set queries as initial state for the ODE solve
        z0 = target_queries.requires_grad_(True)
        
        t = torch.tensor([0.0, 1.0], device=context_latents.device, dtype=context_latents.dtype)
        evolved_state = odeint(self.unified_brain, z0, t, method='dopri5')[-1]
        
        self.unified_brain.current_action = None

        # 3. Final Adaptive Thinking
        return self.norm(self.ttt_layer(evolved_state))
