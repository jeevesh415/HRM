# Human-Vision + Execution Evaluation Spec (Initial)

This document translates project purpose into executable evaluation tracks.

## Purpose Alignment
Target capabilities:
- color and illumination robustness
- depth/geometry continuity
- shadow/reflectance stability
- action-conditioned future consistency
- long-horizon cognitive execution

## Track A: Perception Robustness (implemented baseline)
Command:
```bash
python evaluate_perception.py --config config/vjepa_micro.yaml --seed 42
```

Outputs:
- `color_jitter_latent_l2`
- `brightness_shadow_latent_l2`
- `gaussian_noise_latent_l2`
- `spatial_shift_latent_l2`

Lower is better (more invariant latent representations).

## Track B: World-Model Dynamics (implemented baseline)
Command:
```bash
python evaluate_world_model.py --config config/vjepa_micro.yaml --seed 42
```

Outputs:
- `rollout_drift_l2`
- `trajectory_divergence_l2`
- `max_action_prior`
- `action_prior_entropy`

## Track C: Execution/Cognition (next)
- goal-conditioned planning success@k
- action counterfactual consistency
- uncertainty-aware risk-return tradeoff

## Promotion Rule (Phase-1/2)
A change is promoted only if:
1. no regression in compile/smoke execution,
2. no major degradation in Track A/B metrics for same seed/config,
3. rationale + ablation switch is documented.
