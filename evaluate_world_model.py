"""
World-model evaluation harness for V-JEPA/HRM.

Phase-1 rigorous evaluation focuses on:
  1) rollout drift (latent-state drift over repeated imagination)
  2) action consistency (whether action-conditioned futures are distinct)
  3) calibration-oriented proxy metrics (uncertainty magnitude + confidence proxy)

This script is intentionally lightweight and self-contained so it can be
used early in development before full benchmark infrastructure is added.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List

import torch
import yaml

from models.vjepa.vjepa_model import VJEPA


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EvalManifest:
    timestamp_utc: str
    commit: str
    config_path: str
    seed: int
    device: str
    batch_size: int
    rollout_steps: int
    num_actions: int
    metrics: Dict[str, float]


def get_commit_hash(default: str = "unknown") -> str:
    head = os.path.join(".git", "HEAD")
    if not os.path.exists(head):
        return default
    try:
        with open(head, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        if ref.startswith("ref: "):
            ref_path = os.path.join(".git", ref.split(" ", 1)[1])
            with open(ref_path, "r", encoding="utf-8") as f:
                return f.read().strip()[:12]
        return ref[:12]
    except Exception:
        return default


def latent_rollout(
    model: VJEPA,
    state: torch.Tensor,
    actions: List[torch.Tensor],
) -> List[torch.Tensor]:
    states = [state]
    cur = state
    for action in actions:
        dt = torch.ones(cur.shape[0], device=cur.device)
        cur = model.predictor.physics_engine(cur, dt, action=action)
        states.append(cur)
    return states


def evaluate_metrics(model: VJEPA, device: torch.device, rollout_steps: int, num_actions: int) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        dim = model.value_head[0].in_features
        bs = 1
        seq_len = 16
        z0 = torch.randn(bs, seq_len, dim, device=device)

        actions_a = [torch.randn(bs, seq_len, 128, device=device) for _ in range(rollout_steps)]
        actions_b = [torch.randn(bs, seq_len, 128, device=device) for _ in range(rollout_steps)]

        traj_a = latent_rollout(model, z0, actions_a)
        traj_b = latent_rollout(model, z0, actions_b)

        # 1) rollout drift: average step-to-step displacement in a rollout
        step_drifts = []
        for t in range(1, len(traj_a)):
            step_drifts.append((traj_a[t] - traj_a[t - 1]).pow(2).mean().sqrt().item())
        rollout_drift = float(sum(step_drifts) / max(len(step_drifts), 1))

        # 2) action consistency proxy: trajectories from distinct actions should diverge
        trajectory_divergence = float((traj_a[-1] - traj_b[-1]).pow(2).mean().sqrt().item())

        # 3) planner prior concentration (confidence proxy)
        available_actions = torch.randn(num_actions, 128, device=device)
        pooled = traj_a[-1].mean(dim=1)
        query = model.policy_query_head(pooled).squeeze(0)
        logits = torch.matmul(available_actions, query)
        probs = torch.softmax(logits, dim=0)
        max_prior = float(probs.max().item())
        prior_entropy = float(-(probs * (probs + 1e-9).log()).sum().item())

        return {
            "rollout_drift_l2": rollout_drift,
            "trajectory_divergence_l2": trajectory_divergence,
            "max_action_prior": max_prior,
            "action_prior_entropy": prior_entropy,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA/HRM world-model metrics")
    parser.add_argument("--config", default="config/vjepa_micro.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--num-actions", type=int, default=32)
    parser.add_argument("--save-dir", default="eval_runs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = VJEPA(
        config["encoder"],
        config["predictor"],
        config["training"]["ema_momentum"],
        action_dim=128,
    ).to(device)

    metrics = evaluate_metrics(
        model=model,
        device=device,
        rollout_steps=args.rollout_steps,
        num_actions=args.num_actions,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    manifest = EvalManifest(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        commit=get_commit_hash(),
        config_path=args.config,
        seed=args.seed,
        device=str(device),
        batch_size=1,
        rollout_steps=args.rollout_steps,
        num_actions=args.num_actions,
        metrics=metrics,
    )

    out_path = os.path.join(args.save_dir, f"world_model_eval_{manifest.commit}_{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)

    print("Evaluation complete.")
    print(json.dumps(asdict(manifest), indent=2))


if __name__ == "__main__":
    main()
