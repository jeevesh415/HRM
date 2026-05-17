"""
Adaptive Depth for Test-Time Compute Scaling.

Implements dynamic reasoning depth based on input difficulty:
  - Easy inputs: 1-2 H/L cycles (fast, confident)
  - Hard inputs: up to halt_max_steps cycles (deep reasoning)

This mimics human cognition: familiar patterns get instant recognition,
while novel or confusing patterns trigger deeper analysis.

The Q-head's halt probability serves as the confidence signal.
High confidence = halt early. Low confidence = keep thinking.

Based on:
  - Adaptive Computation Time (Graves, 2016)
  - HRM's own ACT mechanism (extended)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class AdaptiveDepthController(nn.Module):
    """
    Controls reasoning depth based on input difficulty.

    Monitors the Q-head's halt probability and decides when to stop
    reasoning. Implements three regimes:
      1. Fast path: halt after 1 step if very confident
      2. Standard path: halt when Q-head says to (normal ACT)
      3. Deep path: force more steps if uncertainty is high

    Args:
        max_depth: maximum number of reasoning cycles.
        confidence_threshold: halt early if confidence exceeds this.
        uncertainty_threshold: force extra steps if uncertainty exceeds this.
        min_depth: minimum reasoning depth (always at least this many steps).
    """

    def __init__(
        self,
        max_depth: int = 16,
        confidence_threshold: float = 0.95,
        uncertainty_threshold: float = 0.7,
        min_depth: int = 1,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.min_depth = min_depth

    def should_continue(
        self,
        step: int,
        q_halt_logits: torch.Tensor,
        q_continue_logits: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decide whether to continue reasoning for each sample in the batch.

        Args:
            step: current step number (0-indexed).
            q_halt_logits: (bs,) halt logits from Q-head.
            q_continue_logits: (bs,) continue logits from Q-head.
            uncertainty: (bs,) optional uncertainty estimate.

        Returns:
            should_halt: (bs,) boolean tensor — True means HALT.
            info: dictionary with diagnostic information.
        """
        import torch.nn.functional as F

        bs = q_halt_logits.shape[0]
        device = q_halt_logits.device

        # Compute halt probability
        halt_prob = torch.sigmoid(q_halt_logits)
        continue_prob = torch.sigmoid(q_continue_logits)
        confidence = halt_prob  # confidence in "halt" decision

        # Base halt condition: Q-head says halt AND minimum depth reached
        base_halt = (q_halt_logits > q_continue_logits) & (step >= self.min_depth)

        # Fast path: very confident → halt immediately
        fast_halt = (confidence > self.confidence_threshold) & (step >= 1)

        # Deep path: very uncertain → force more steps
        if uncertainty is not None:
            force_continue = uncertainty > self.uncertainty_threshold
        else:
            # Use inverse confidence as uncertainty proxy
            force_continue = (1 - confidence) > self.uncertainty_threshold

        force_continue = force_continue & (step < self.max_depth - 1)

        # Final decision
        should_halt = (base_halt | fast_halt) & ~force_continue

        # Always halt at max depth
        should_halt = should_halt | (step >= self.max_depth - 1)

        info = {
            "confidence": confidence,
            "halt_prob": halt_prob,
            "step": step,
            "fast_halted": (fast_halt & should_halt).sum(),
            "deep_forced": force_continue.sum(),
        }

        return should_halt, info


class AdaptiveDepthWrapper(nn.Module):
    """
    Wraps an HRM model with adaptive depth control.

    Instead of running a fixed number of H/L cycles, this wrapper
    monitors the model's confidence and dynamically adjusts the
    reasoning depth.

    Args:
        model: the inner HRM model (HierarchicalReasoningModel_ACTV1_Inner).
        max_depth: maximum reasoning depth.
        confidence_threshold: early halt threshold.
        uncertainty_threshold: forced continuation threshold.
    """

    def __init__(
        self,
        model: nn.Module,
        max_depth: int = 16,
        confidence_threshold: float = 0.95,
        uncertainty_threshold: float = 0.7,
    ):
        super().__init__()
        self.model = model
        self.depth_controller = AdaptiveDepthController(
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold,
        )

    def forward(
        self,
        carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple:
        """
        Forward pass with adaptive depth.

        Runs the model iteratively, checking confidence at each step.
        Halts early for confident samples, continues for uncertain ones.
        """
        # Initialize
        new_inner_carry = self.model.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # Run inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.model.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1

            # Adaptive depth decision
            halted, depth_info = self.depth_controller.should_continue(
                step=new_steps[0].item(),  # use first sample's step as reference
                q_halt_logits=q_halt_logits,
                q_continue_logits=q_continue_logits,
            )

            outputs["depth_info"] = depth_info

        return (
            type(carry)(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
