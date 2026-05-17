# Rigorous Development Protocol (Phase-1)

This protocol defines minimum rigor gates for model and planner changes.

## Gate A — Sanity / Determinism
1. Python compile sanity for repository modules.
2. Seeded execution for evaluation scripts.
3. Basic tensor-shape and NaN safety in smoke runs.

## Gate B — World-model Metrics
Run:

```bash
python evaluate_world_model.py --config config/vjepa_micro.yaml --seed 42
```

Required outputs:
- `rollout_drift_l2`
- `trajectory_divergence_l2`
- `max_action_prior`
- `action_prior_entropy`

All metrics are persisted as a JSON run manifest in `eval_runs/`.

## Gate C — Change Promotion
Any feature PR must include:
1. Before/after metric table (same config + seed).
2. Ablation switch (enable/disable path).
3. Short rationale for metric movement.

## Notes
- This is a phase-1 lightweight protocol and will be extended with calibration
  and long-horizon benchmark suites in subsequent iterations.
