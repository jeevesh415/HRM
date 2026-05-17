"""
Phase-2 perception robustness evaluation for HRM + V-JEPA.

Focus:
- color robustness
- brightness/shadow robustness
- noise robustness
- geometric perturbation robustness

This uses latent consistency between original and perturbed clips as an
early proxy for perceptual invariance before task-specific benchmarks.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict

import torch
import yaml

from models.vjepa.vjepa_model import VJEPA


def apply_perturbation(video: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "color_jitter":
        scale = torch.tensor([1.1, 0.9, 1.05], device=video.device).view(1, 1, 3, 1, 1)
        return (video * scale).clamp(-3.0, 3.0)
    if mode == "brightness_shadow":
        return (video * 0.7).clamp(-3.0, 3.0)
    if mode == "gaussian_noise":
        return video + 0.05 * torch.randn_like(video)
    if mode == "spatial_shift":
        return torch.roll(video, shifts=2, dims=-1)
    raise ValueError(f"Unknown perturbation mode: {mode}")


def latent_consistency(model: VJEPA, video: torch.Tensor, perturbed: torch.Tensor) -> float:
    with torch.no_grad():
        z_ref = model.context_encoder(video)
        z_alt = model.context_encoder(perturbed)
        return float((z_ref - z_alt).pow(2).mean().sqrt().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/vjepa_micro.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="eval_runs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = VJEPA(
        cfg["encoder"],
        cfg["predictor"],
        cfg["training"]["ema_momentum"],
        action_dim=128,
    ).to(device).eval()

    # synthetic clip for deterministic smoke-evaluation
    bs, t, c, h, w = 1, cfg["encoder"]["max_t"] * cfg["encoder"]["patch_size"][0], 3, cfg["encoder"]["img_size"], cfg["encoder"]["img_size"]
    video = torch.randn(bs, t, c, h, w, device=device)

    metrics: Dict[str, float] = {}
    for mode in ["color_jitter", "brightness_shadow", "gaussian_noise", "spatial_shift"]:
        pert = apply_perturbation(video, mode)
        metrics[f"{mode}_latent_l2"] = latent_consistency(model, video, pert)

    os.makedirs(args.save_dir, exist_ok=True)
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "seed": args.seed,
        "device": str(device),
        "metrics": metrics,
    }
    path = os.path.join(args.save_dir, f"perception_eval_seed_{args.seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

