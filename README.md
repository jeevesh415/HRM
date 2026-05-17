# Visual Execution Model (VEM)

A single integrated framework for **continuous-time world modeling** from video.

Visual Execution Model (VEM) unifies hierarchical reasoning, predictive representation learning, dynamics, geometry, planning, uncertainty, and multimodal grounding inside one model stack (not separate models).

---

## Purpose
Build a practical foundation for models that can:
1. Learn physical regularities directly from raw video,
2. Reason over future latent trajectories,
3. Support intervention-aware planning in latent space.

This repository is organized as one unified model pipeline with scalable sizes and optional modules, so every capability is part of the same framework and execution graph.

## Vision
Our vision is a model that develops robust **intuitive physics** (e.g., continuity, object permanence, motion consistency, and causal effects of actions) by combining representation learning, geometric priors, and dynamics-aware objectives.

## Goal
Deliver a scalable and analyzable training stack that can evolve from micro-scale experiments to large configurations (including 10B-class settings) while preserving:
- modularity,
- mathematical interpretability,
- and reproducible workflow.

---

## Technical Architecture (Concept Map)

### 1) Spatio-Temporal Representation (Vision Encoder)
- **3D patch embedding** over `(T, H, W)` video volumes.
- **3D-RoPE** positional encoding in time-height-width coordinates.
- ViT-style latent tokenization for downstream predictive modeling.

### 2) Geometric Inductive Biases
- **Lie-group / equivariance-oriented layers** for transformation-aware latent features.
- **Stiefel-manifold style orthogonality constraints/projections** to stabilize relational geometry.
- **Proper SE(3)-inspired processing** for physically meaningful transformations.

### 3) Continuous-Time Latent Dynamics
- **Hamiltonian-style latent dynamics** components.
- **Neural ODE adjoint** pathway (`torchdiffeq`) for memory-efficient continuous-time learning.
- **Symplectic integration path** for structure-preserving latent evolution at inference-style rollout.

### 4) Hierarchical Predictive Reasoning
- High/Low cycle interaction (`H_cycles`, `L_cycles`) for iterative latent refinement.
- Predictive coding flavor with top-down influence and bottom-up correction pressure.
- Adaptive compute hooks (e.g., ACT/depth controller) for confidence-aware depth.

### 5) World Rendering and Latent Scene Composition
- **Latent Gaussian Splatting** path for explicit scene primitive aggregation.
- NeRF-inspired latent rendering concepts for geometry/appearance reasoning.

### 6) Latent Planning & Decision Support
- **Latent MCTS** module for action-conditioned future evaluation.
- Value estimation head for ranking latent future states.

### 7) Multi-Modal and Robustness Extensions
- Hooks for **audio** and **tactile/proprioceptive** grounding.
- **Uncertainty estimation**, **information bottleneck**, **topology-aware**, and **spectral** auxiliary modules.

### 8) Training Stack
- **VICReg** objective (invariance + variance/covariance regularization).
- Spatio-temporal masking regime.
- Optimizer backends: **AdamW**, **Muon**, or **Hybrid Muon+AdamW**.
- EMA target encoder for stable JEPA-style targets.

---

## Repository Workflow (Single Framework)

### Configurations
- **Micro scale profile (same model, small size)**: `config/vjepa_micro.yaml`
- **Large scale profile (same model, 10B-class size target)**: `config/vjepa_10b.yaml`

### Training Entrypoint
```bash
python vjepa_train.py --config config/vjepa_micro.yaml
# or
python vjepa_train.py --config config/vjepa_10b.yaml
```

`vjepa_train.py` accepts `--config` and loads runtime behavior from YAML. Both configs run the same Visual Execution Model framework at different scales.  
`training.epochs` can be set in YAML (defaults to `100` if omitted).

### Practical Notes
- Place video files in `data/` for training.
- If `data/` is absent, the script attempts to create it and generate a small synthetic test video via `ffmpeg`.
- For phase-1 rigorous world-model checks, run:
  `python evaluate_world_model.py --config config/vjepa_micro.yaml --seed 42`
  (saves JSON manifests in `eval_runs/`).
- For perception robustness checks (color/shadow/noise/shift), run:
  `python evaluate_perception.py --config config/vjepa_micro.yaml --seed 42`
  (saves JSON manifests in `eval_runs/`).

### Final Execution Checklist (Do This)
1. `python -m compileall -q .`
2. `python evaluate_world_model.py --config config/vjepa_micro.yaml --seed 42`
3. `python evaluate_perception.py --config config/vjepa_micro.yaml --seed 42`
4. `python vjepa_train.py --config config/vjepa_micro.yaml` (with real videos in `data/`, or with `ffmpeg` installed)

---

## Roadmap Direction
- Stronger experiment tracking and benchmark reports.
- Expanded multimodal pretraining/evaluation.
- Systematic ablations on dynamics engines (ODE vs. flow matching vs. symplectic rollout).
- Better reproducibility packaging for large-scale distributed runs.

---

This project is focused on pushing **hierarchical physical reasoning** toward robust, scalable world models with clear technical structure and research extensibility.
