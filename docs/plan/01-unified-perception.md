# /plan: Unified Continuous Perception Functional (UCPF)

## 1. Objective
Refactor the modular, sequential architecture of `models/vjepa/predictor.py` into a unified continuous functional. This involves "folding" the physics engine (`HRMPhysicsODE`) and the Gaussian splatting renderer (`LatentGaussianSplatting`) into a single **Hamiltonian Neural ODE** vector field.

## 2. Structural Changes
- **Core File:** `models/vjepa/predictor.py`
- **Supporting Files:** 
  - `models/vjepa/physics_engine.py` (Decommission modular `ContinuousTimeHRM`)
  - `models/vjepa/gaussian_splatting.py` (Extract parameters for unified functional)

## 3. Implementation Steps

### Phase 1: The Unified Vector Field
1.  **Define `UnifiedVectorField` Class:**
    - Inherit from `nn.Module`.
    - Internalize `energy_net` from `HRMPhysicsODE`.
    - Internalize `gaussian_encoder` and `latent_to_3d` from `LatentGaussianSplatting`.
    - Implement `forward(t, z)` to compute:
      - $\nabla_z H(z, \text{visual_features})$
      - Gaussian splatting weights based on $z$ (interpreted as a manifold position).
      - Return the structured Hamiltonian gradient modified by the visual rendering density.

### Phase 2: Predictor Refactor
1.  **Initialize `UnifiedVectorField`** in `VJEPAPredictorInner.__init__`.
2.  **Remove sequential blocks** from `VJEPAPredictorInner.forward`:
    - Delete `self.physics_engine` call.
    - Delete `self.gaussian_splatting` call.
    - Delete `self.ray_marcher` call.
3.  **Implement Single ODE Call:**
    - Use `odeint_adjoint` to solve the `UnifiedVectorField`.
    - The output of the ODE integration *is* the final unified perception.

### Phase 3: Integration & Clean-up
1.  **Decommission Modular Files:** Mark `physics_engine.py` and `gaussian_splatting.py` as legacy or remove if no longer needed.
2.  **Update Configs:** Ensure `vjepa_micro.yaml` points to the new unified architecture.

## 4. Verification Plan
- **Verification Gate 1 (Static):** Ensure `VJEPAPredictorInner.forward` contains exactly one major execution block (the ODE solver).
- **Verification Gate 2 (Dynamic):** Run `vjepa_train.py` with the new architecture to confirm convergence and visual perception smoothness.
- **Anti-Rationalization Check:** Confirm that no `+` operations exist between physics and rendering; they must be terms in the same vector field.

## 5. Traceability
- **LiOps Log:** Every change will be recorded in `logs/research_trace.jsonl`.
- **Memory Palace:** The final architecture will be "stored" in the `Technical_Architecture` Room.
