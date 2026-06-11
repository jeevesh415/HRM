# /plan: V-JEPA 2.1 Build & Commit

## Phase 1: Encoder Upgrade (`vit.py`)
1.  **Modify `VisionEncoder.forward`**:
    - Add logic to collect and return activations from a list of `target_layers` (e.g., layers 4, 8, 12).
    - Ensure 3D-RoPE is applied consistently across all levels.

## Phase 2: Predictor Unification (`predictor.py`)
1.  **Refactor `UnifiedVectorField`**:
    - Internalize `LieGroupEquivariantLayer`.
    - Internalize `HolographicMemory`.
    - Folding hierarchical loops (`H_cycles`, `L_cycles`) into the vector field logic.
    - Ensure the ODE output is the **final grounded prediction**.

## Phase 3: Model Wrapper & Training (`vjepa_model.py`, `vjepa_train.py`)
1.  **Update `VJEPA.forward`**:
    - Generate targets for **all** token positions (Dense Loss).
    - Handle multi-layer activations for Deep Self-Supervision.
2.  **Update Loss Calculation**:
    - Aggregate loss across all supervised layers and token positions.

## Phase 4: Verification & Git Push
1.  **Static Analysis**: Double-check all tensor shapes and autograd flows.
2.  **Commit**: `git add .` and `git commit -m "feat: Upgrade to V-JEPA 2.1 with Total Architectural Unification"`
3.  **Push**: `git push origin feature/unified-perception-functional`
