import torch
from torch import nn
import torch.nn.functional as F
import copy
from typing import Dict, Tuple, Optional

from models.vjepa.vit import VisionEncoder
from models.vjepa.predictor import VJEPAPredictorInner
from models.vjepa.utils import apply_mask
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config

class VJEPA(nn.Module):
    """
    Unified V-JEPA Model with HRM-ODE Predictor and Holographic Memory.
    Designed for 10B parameter physical world modeling.
    """
    def __init__(self, 
                 encoder_config: dict,
                 predictor_config: dict,
                 ema_momentum: float = 0.996):
        super().__init__()
        
        # 1. Context Encoder (Online)
        self.context_encoder = VisionEncoder(**encoder_config)
        
        # 2. Target Encoder (EMA) - The 'Ground Truth' Generator
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.ema_momentum = ema_momentum
        
        # 3. Predictor (The HRM-ODE Brain)
        # Convert dict to HRM Config object
        p_cfg = HierarchicalReasoningModel_ACTV1Config(**predictor_config)
        self.predictor = VJEPAPredictorInner(
            dim=p_cfg.hidden_size,
            num_heads=p_cfg.num_heads,
            expansion=p_cfg.expansion,
            h_cycles=p_cfg.H_cycles,
            l_cycles=p_cfg.L_cycles
        )
        
        # 4. Halting/ACT Head (inherited from HRM logic)
        self.q_head = nn.Linear(p_cfg.hidden_size, 2)

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update for the target network."""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.ema_momentum + param_q.data * (1.0 - self.ema_momentum)

    def forward(self, batch: Dict[str, torch.Tensor]):
        video = batch["video"] # (bs, T, C, H, W)
        mask = batch["mask"]   # (bs, seq_len)
        delta_t = batch["delta_t"] # (bs, 1)
        
        # 1. Generate Target Latents (Full Video, No Gradients)
        with torch.no_grad():
            target_latents = self.target_encoder(video) # (bs, seq_len, D)
            
        # 2. Generate Context Latents (Masked Video)
        # For simplicity, we encode full video and then mask in latent space
        # Official V-JEPA only encodes visible patches for efficiency.
        all_latents = self.context_encoder(video)
        
        # Extract visible patches for context
        # Extract masked patches as targets
        context_latents, target_latents_masked = apply_mask(all_latents, mask)
        _, target_truth_masked = apply_mask(target_latents, mask)

        # 3. Predictor Forward (Continuous-Time Reasoning)
        # We need the 3D cos_sin for the masked positions
        # For now, we pass None and let it use its internal logic or positional embeddings
        cos_sin = self.context_encoder.rope(self.context_encoder.max_t, self.context_encoder.max_h, self.context_encoder.max_w)
        
        # Predict masked latents
        # target_queries are the positional embeddings of the masked patches
        # Here we just use the masked context latents (which would be zero/mask tokens in a real scenario)
        predicted_latents = self.predictor(
            context_latents=context_latents,
            target_queries=target_latents_masked, # This should ideally be positional tokens
            cos_sin=cos_sin,
            delta_t=delta_t
        )

        return {
            "predicted": predicted_latents,
            "target": target_truth_masked,
            "all_context": all_latents # For VICReg variance/covariance
        }
