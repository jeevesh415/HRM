import torch
from torch import nn
import torch.nn.functional as F
import copy
from typing import Dict, Tuple, Optional

from models.vjepa.vit import VisionEncoder
from models.vjepa.predictor import VJEPAPredictorInner
from models.vjepa.utils import apply_mask
from models.adaptive_depth import AdaptiveDepthController

class VJEPA(nn.Module):
    """
    Unified V-JEPA Model with HRM-ODE Predictor and Holographic Memory.
    Designed for 10B parameter physical world modeling.

    Enhancements over base:
      - 3D Gaussian Splatting latent renderer
      - Flow Matching dynamics engine
      - Symplectic integrator for energy conservation
      - Muon optimizer support (configured externally)
    """
    def __init__(self, 
                 encoder_config: dict,
                 predictor_config: dict,
                 ema_momentum: float = 0.996,
                 action_dim: int = 128):
        super().__init__()
        
        # 1. Context Encoder (Online)
        self.context_encoder = VisionEncoder(**encoder_config)
        
        # 2. Target Encoder (EMA) - The 'Ground Truth' Generator
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.ema_momentum = ema_momentum
        
        # 3. Predictor (The HRM-ODE Brain)
        self.predictor = VJEPAPredictorInner(
            dim=predictor_config["hidden_size"],
            num_heads=predictor_config["num_heads"],
            expansion=predictor_config["expansion"],
            h_cycles=predictor_config["H_cycles"],
            l_cycles=predictor_config["L_cycles"],
            action_dim=action_dim,
            use_gaussian_splatting=predictor_config.get("use_gaussian_splatting", True),
            use_flow_matching=predictor_config.get("use_flow_matching", True),
            use_symplectic=predictor_config.get("use_symplectic", True),
            num_gaussians=predictor_config.get("num_gaussians", 256),
        )
        
        # 4. Halting/ACT Head (inherited from HRM logic)
        self.q_head = nn.Linear(predictor_config["hidden_size"], 2)

        # 5. Value Head for Latent Planning (MCTS)
        self.value_head = nn.Sequential(
            nn.Linear(predictor_config["hidden_size"], predictor_config["hidden_size"]),
            nn.SiLU(),
            nn.Linear(predictor_config["hidden_size"], 1)
        )

        # 6. Adaptive depth controller for test-time compute scaling
        self.depth_controller = AdaptiveDepthController(
            max_depth=predictor_config.get("halt_max_steps", 8),
            confidence_threshold=predictor_config.get("confidence_threshold", 0.95),
            uncertainty_threshold=predictor_config.get("uncertainty_threshold", 0.7),
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update for the target network."""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.ema_momentum + param_q.data * (1.0 - self.ema_momentum)

    def forward(self, batch: Dict[str, torch.Tensor]):
        video = batch["video"] # (bs, T, C, H, W)
        mask = batch["mask"]   # (bs, seq_len)
        delta_t = batch.get("delta_t", torch.ones(video.shape[0], 1, device=video.device)) 
        action = batch.get("action", None)
        
        # 1. Generate Target Latents (Full Video, No Gradients)
        with torch.no_grad():
            target_latents = self.target_encoder(video) # (bs, seq_len, D)
            
        # 2. Generate Context Latents (Masked Video)
        all_latents = self.context_encoder(video)
        
        # Extract visible patches for context
        # Extract masked patches as targets
        context_latents, target_latents_masked = apply_mask(all_latents, mask)
        _, target_truth_masked = apply_mask(target_latents, mask)

        # 3. Predictor Forward (Continuous-Time Reasoning)
        full_cos, full_sin = self.context_encoder.rope(self.context_encoder.max_t, self.context_encoder.max_h, self.context_encoder.max_w)
        
        # Index cos_sin for masked positions
        if mask.ndim == 1:
            masked_cos = full_cos[mask]
            masked_sin = full_sin[mask]
        else:
            masked_cos = torch.stack([full_cos[m_i] for m_i in mask], dim=0)
            masked_sin = torch.stack([full_sin[m_i] for m_i in mask], dim=0)

        masked_cos_sin = (masked_cos, masked_sin)

        # Predict masked latents
        predicted_latents = self.predictor(
            context_latents=context_latents,
            target_queries=target_latents_masked,
            cos_sin=masked_cos_sin,
            delta_t=delta_t,
            action=action,
            audio_features=batch.get("audio", None),
            tactile_features=batch.get("tactile", None),
        )

        # Value estimation of the predicted future
        value = self.value_head(predicted_latents.mean(dim=1))

        return {
            "predicted": predicted_latents,
            "target": target_truth_masked,
            "all_context": all_latents,
            "value": value
        }
