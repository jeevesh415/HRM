# Hierarchical Reasoning Model (HRM) + V-JEPA

An advanced research codebase for **continuous-time world modeling** from video, combining:
- **HRM** (hierarchical latent reasoning),
- **V-JEPA** (self-supervised predictive representation learning),
- and mathematically grounded modules for dynamics, geometry, planning, and uncertainty.

---

## Purpose
Build a practical foundation for models that can:
1. Learn physical regularities directly from raw video,
2. Reason over future latent trajectories,
3. Support intervention-aware planning in latent space.

This repository transitions from discrete puzzle-style reasoning to **continuous latent dynamics** with explicit architectural support for long-horizon prediction.

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

## Repository Workflow

### Configurations
- **Micro / local iteration**: `config/vjepa_micro.yaml`
- **Large-scale profile**: `config/vjepa_10b.yaml`

### Training Entrypoint
```bash
python vjepa_train.py --config config/vjepa_micro.yaml
# or
python vjepa_train.py --config config/vjepa_10b.yaml
```

`vjepa_train.py` accepts `--config` and loads runtime behavior from YAML.  
`training.epochs` can be set in YAML (defaults to `100` if omitted).

### Practical Notes
- Place video files in `data/` for training.
- If `data/` is absent, the script attempts to create it and generate a small synthetic test video via `ffmpeg`.

---

## Roadmap Direction
- Stronger experiment tracking and benchmark reports.
- Expanded multimodal pretraining/evaluation.
- Systematic ablations on dynamics engines (ODE vs. flow matching vs. symplectic rollout).
- Better reproducibility packaging for large-scale distributed runs.

---

This project is focused on pushing **hierarchical physical reasoning** toward robust, scalable world models with clear technical structure and research extensibility.
