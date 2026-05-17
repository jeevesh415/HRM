# Frontier Capability Gap Analysis and Implementation Plan

## Scope
This document compares:
1. Existing repository capabilities.
2. State-of-the-art (frontier labs + top academic trends) capability expectations.
3. Immediate implementation decisions.

## Comparison Matrix

| Area | Existing in repo | Frontier expectation | Gap | Action |
|---|---|---|---|---|
| Latent world modeling | V-JEPA-style masked latent prediction and EMA target path | Long-horizon stable latent rollouts with robust eval | Partial | Add dedicated world-model evaluation harness (next step) |
| Continuous-time dynamics | Hamiltonian/ODE/symplectic modules present | Quantitative invariance + long-horizon stability metrics | Missing benchmarks | Add metrics + ablations (next step) |
| Latent planning (MCTS) | MCTS scaffold existed with placeholder action priors | Policy-informed planning priors and uncertainty-aware scoring | Prior quality gap | Implemented policy-query action priors in MCTS |
| Uncertainty | Uncertainty module present | Planning/calibration integration | Partial | Integrate uncertainty into planner scoring (planned) |
| Multimodal grounding | Audio/tactile hooks + cross-modal attention | Curriculum and modality-drop robustness metrics | Partial | Add modality-drop ablations (planned) |
| Reproducible workflow | Configurable training entrypoint | Evaluation protocol + run manifests + acceptance gates | Partial | Add benchmark specs and run manifests (planned) |

## Implemented in this change

### 1) MCTS action prior upgrade
Previously, planning used a placeholder prior logits tensor. We replaced that with a learned action-prior mechanism:
- A new `policy_query_head` in `VJEPA` maps latent state to an action-space query vector.
- MCTS computes action priors from dot-product similarity between candidate actions and the learned query.
- This upgrades search from uniform/placeholder priors to model-informed priors.

## Why this is prioritized first
Planning quality depends heavily on action priors. Replacing placeholder priors is a high-leverage improvement that directly improves practical controllable rollout search.

## Next technical steps (ordered)
1. Add `evaluate_world_model.py` with rollout drift, action-consistency, and calibration metrics.
2. Wire uncertainty estimates into PUCT scoring (risk-aware planning).
3. Add ablation configs for dynamics engines and multimodal drop robustness.
4. Define acceptance thresholds for promotion of each advanced module.
