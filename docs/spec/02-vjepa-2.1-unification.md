# /spec: V-JEPA 2.1 Frontier Integration & Total Unification

## 1. Problem Statement
The current codebase implements a "Modular V-JEPA" (v1.0-style) with a placeholder "Unified" predictor. This configuration relies on sparse supervision (masked tokens only) and lacks the hierarchical grounding required for a "Fantastic Scientist" level of world modeling.

## 2. Objective
Upgrade the repository to **V-JEPA 2.1** standards and complete the **Total Unification** of the architecture. This transforms the model from a puzzle-solver into a **Dense World Model**.

## 3. Technical Requirements (V-JEPA 2.1)
- **Dense Predictive Loss:** Update the training objective to supervise **all tokens** (visible context + masked targets). This prevents "shortcut" learning and ensures universal grounding.
- **Deep Self-Supervision:** Modify the `VisionEncoder` to return activations from intermediate layers. Apply VICReg loss across multiple depths to capture both low-level geometry (depth/height) and high-level semantics.
- **Predictor Unification (De-Lego-fication):** Merge `LieGroupEquivariantLayer`, `HolographicMemory`, and `HierarchicalReasoning` directly into the `UnifiedVectorField`. The ODE solver must be the sole engine of thought.

## 4. Architectural Map
- **Encoder:** `models/vjepa/vit.py` -> Support intermediate feature extraction.
- **Predictor:** `models/vjepa/predictor.py` -> The "Integrated Brain" (Hamiltonian + Memory + Refinement).
- **Model Wrapper:** `models/vjepa/vjepa_model.py` -> Implement 2.1 forward pass (Dense Loss).
- **Loss:** `models/vjepa/losses.py` -> Support hierarchical supervision.

## 5. Verification Gates
- **Gate 1 (Static):** Codebase must be free of sequential processing blocks between sensory input and predictive output.
- **Gate 2 (Traceability):** Every 2.1 feature must be logged in `research_trace.jsonl`.
- **Gate 3 (Commit):** All changes must be pushed to `feature/unified-perception-functional` with high-signal technical documentation.
