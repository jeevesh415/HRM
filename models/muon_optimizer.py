"""
Muon Optimizer: Muon Is an Optimizer Using Newton-Schulz.

Achieves ~2x computational efficiency vs AdamW for LLM training by using
Newton-Schulz iteration for matrix orthogonalization instead of second-moment
estimates. This replaces AdamW's per-parameter variance tracking with a
geometrically principled approach that keeps gradient updates well-conditioned.

Based on: "Muon is Scalable for LLM Training" (Liu et al., 2025)
and the Moonshot AI implementation (Feb 2025).

Key innovations:
  - Newton-Schulz orthogonalization of momentum buffer (5 iterations)
  - Nesterov-style momentum for faster convergence
  - Decoupled weight decay (same as AdamW)
  - Per-parameter scale adjustment for numerical stability
  - ZeRO-1 style distributed gradient all-reduce
"""

import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
from typing import Optional, List, Dict, Any


class Muon(Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization.

    Works best for 2D+ parameters (weight matrices). For 1D parameters
    (biases, layernorm weights), falls back to sign-based updates.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate (default: 0.02).
        momentum: momentum coefficient (default: 0.95).
        weight_decay: decoupled weight decay (default: 0.1).
        nesterov: use Nesterov-style momentum (default: True).
        ns_steps: number of Newton-Schulz iterations (default: 5).
        world_size: number of distributed workers for gradient sync (default: 1).
        rank: local rank for distributed training (default: 0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        ns_steps: int = 5,
        world_size: int = 1,
        rank: int = 0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self.world_size = world_size
        self.rank = rank

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Optionally sync gradients across distributed workers
        all_params = []
        for group in self.param_groups:
            all_params.extend(group["params"])
        if self.world_size > 1:
            self._distributed_allreduce_grads(all_params, self.world_size, self.rank)

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Update momentum buffer: buf = momentum * buf + grad
                buf.mul_(momentum).add_(grad)

                # Compute update direction
                if nesterov:
                    # Nesterov: update = grad + momentum * buf
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf.clone()

                # Apply Newton-Schulz orthogonalization for matrix params
                if update.ndim >= 2:
                    update = self._newton_schulz_orthogonalize(update, ns_steps)
                else:
                    # For 1D params (biases, norms), use sign-based update
                    update = update.sign()

                # Decoupled weight decay (applied before the gradient step)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # Apply update with scale adjustment for stability
                # Scale by sqrt(max(rows, cols)) to keep updates well-conditioned
                if update.ndim >= 2:
                    scale = max(update.shape[0], update.shape[1]) ** 0.5
                    update = update / scale

                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    @torch.no_grad()
    def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Newton-Schulz iteration to approximate (G @ G^T)^{-1/2} @ G.

        This orthogonalizes the gradient matrix, ensuring that the update
        directions are well-conditioned and isotropic. The iteration uses
        the recurrence X_{k+1} = (3*X_k - X_k @ X_k^T @ X_k) / 2 which
        converges to the orthogonal factor of G.

        Args:
            G: gradient tensor of shape (..., rows, cols).
            steps: number of Newton-Schulz iterations (default: 5).

        Returns:
            Orthogonalized gradient of the same shape.
        """
        # Normalize to avoid numerical issues
        G_norm = G.norm()
        if G_norm < 1e-7:
            return G
        X = G / G_norm

        # Newton-Schulz iteration: X = (3X - X X^T X) / 2
        # This converges to the orthogonal factor (polar decomposition)
        for _ in range(steps):
            A = X @ X.transpose(-2, -1)
            X = (3 * X - A @ X) / 2

        # Scale back to original norm
        return X * G_norm

    @staticmethod
    @torch.no_grad()
    def _distributed_allreduce_grads(
        params: List[torch.Tensor],
        world_size: int,
        rank: int,
    ) -> None:
        """
        ZeRO-1 style gradient all-reduce.

        Averages gradients across all distributed workers before the
        optimizer step. This is equivalent to ZeRO Stage 1 where
        optimizer states are partitioned but gradients are synchronized.

        Args:
            params: list of parameters whose gradients to synchronize.
            world_size: number of distributed workers.
            rank: local rank (unused here but available for partitioning).
        """
        if world_size <= 1:
            return
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
