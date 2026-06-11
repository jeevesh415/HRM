# HRM Project Instructions

## Project Vision & Philosophy
This project is an advanced research codebase for **continuous-time world modeling from video**. It rejects discrete, box-based or frame-based reasoning in favor of a holistic system that processes the world as continuous curves, colors, depths, and heights—mimicking human visual perception.

**Core Technical Tenets:**
- **V-JEPA Pattern:** Self-supervised predictive learning with spatio-temporal masking.
- **Hierarchical Reasoning (HRM):** Integrated conceptual framework, not a modular "Lego-block" assembly.
- **Continuous Dynamics:** Heavy reliance on PyTorch, `torchdiffeq` for Neural ODEs, and Symplectic integration for physically consistent latent dynamics.
- **Geometric Biases:** Use of Lie-groups and Stiefel-manifolds to ensure latent features respect physical geometry.

## Workflow Rules
- **Branching:** NEVER commit directly to `main`. Always create an isolated feature branch for any task (e.g., `feature/research-concept-x`).
- **Commits:** Write clear, technical commit messages describing the "why" and "how" of architectural changes.
- **Testing:** Verify all changes through project scripts like `vjepa_train.py` or `evaluate.py` where applicable.
- **State-of-the-Art:** Always prioritize the most advanced, mathematically grounded implementation over simpler "industry standard" hacks.
