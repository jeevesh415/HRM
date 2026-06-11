# /spec: Unified Continuous Perception Functional (UCPF)

## 1. Problem Statement
The current `VJEPAPredictorInner` implementation follows a "Lego-block" architecture. It processes geometry, dynamics, and rendering as discrete, sequential modules (`Result = A + B + C`). This creates artificial "seams" and violates the vision of a holistic, human-like perception where depth, height, and color are inseparable facets of a single continuous world state.

## 2. Objective
Transition from a **Sequential Pipeline** to a **Unified Neural Functional**. We will redefine the latent world state as a continuous manifold where physical evolution and visual perception are governed by a single Hamiltonian Energy Functional ($H$).

## 3. Technical Requirements (The "Unification" Math)
- **State Definition:** The world state $z$ will be defined as a point on a **Stiefel Manifold** $V_k(\mathbb{R}^n)$, ensuring internal geometric consistency (orthogonality) is an inherent property, not an added "block."
- **Dynamics-Perception Integration:** Instead of `physics_engine` followed by `ray_marcher`, we will implement a single **Neural ODE** where the vector field $f(z, t)$ is the gradient of a unified energy function: 
  $$\frac{dz}{dt} = \mathbf{J} \nabla_z H(z, \text{visual_features})$$
  This ensures that "moving through time" and "resolving visual depth" are the same mathematical operation.
- **Continuous Rendering:** Replace discrete Gaussian Splatting with a **Volumetric Latent Field** that is sampled continuously during the ODE integration, allowing for "infinite resolution" perception of heights and depths.

## 4. Verification Gates
- **Mathematical Proof:** The new `forward` pass must be representable as a single `torchdiffeq.odeint` call where all "Lego blocks" are folded into the vector field function.
- **Anti-Rationalization Check:** 
  - *Gate:* Are we still adding modules? 
  - *Status:* **NO.** Every concept must be a term within the energy functional $H$.
- **Performance:** Evaluation on the `puzzle_dataset.py` must show smoother latent trajectories (lower curvature) compared to the current modular version.

## 5. Risk Assessment ($\rho$)
- **Complexity:** Higher mathematical overhead for the loss function.
- **Compute:** Neural ODE integration steps may increase training time on Termux; will mitigate using adaptive step-sizes.
