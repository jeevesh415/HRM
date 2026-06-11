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

        # 5b. Policy query head for action-prior scoring in latent MCTS.
        # Produces an action-space query vector that can be matched against
        # candidate action vectors via dot-product similarity.
        self.policy_query_head = nn.Sequential(
            nn.Linear(predictor_config["hidden_size"], predictor_config["hidden_size"]),
            nn.SiLU(),
            nn.Linear(predictor_config["hidden_size"], action_dim)
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
        
        # 1. Generate Target Latents (Dense & Hierarchical)
        # 2.1 Frontier: Supervise intermediate layers for deep self-supervision
        target_layers = [4, 8, 12] if self.context_encoder.max_t > 4 else [0]
        with torch.no_grad():
            target_all_layers = self.target_encoder(video, return_layers=target_layers) 
            target_final = target_all_layers[-1]
            
        # 2. Generate Context Latents (Online Encoder)
        all_latents = self.context_encoder(video)
        
        # 3. Predictor Forward (Total Unification)
        full_cos, full_sin = self.context_encoder.rope(self.context_encoder.max_t, self.context_encoder.max_h, self.context_encoder.max_w)
        masked_cos_sin = (full_cos, full_sin) # In 2.1, we often predict all tokens

        # Predict all tokens (Dense Prediction)
        predicted_latents = self.predictor(
            context_latents=all_latents, # Dense context
            target_queries=all_latents,  # Predict everything
            cos_sin=masked_cos_sin,
            delta_t=delta_t,
            action=action,
            audio_features=batch.get("audio", None),
            tactile_features=batch.get("tactile", None),
        )

        return {
            "predicted": predicted_latents,
            "target": target_final,
            "all_targets": target_all_layers,
            "all_context": all_latents,
        }
