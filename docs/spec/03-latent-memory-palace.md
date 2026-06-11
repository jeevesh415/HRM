# /spec: Latent Memory Palace (LMP) Integration

## 1. Objective
Replace the flat `HolographicMemory` with a **Latent Memory Palace (LMP)**. This provides a hierarchical, long-range memory structure that is fully differentiable and integrated into the `UnifiedVectorField`.

## 2. Hierarchical Architecture
The LMP is structured as a 3-tier latent manifold:
- **Wings ($W$):** Global semantic themes (Top-level embeddings).
- **Rooms ($R$):** Contextual subspaces (Middle-level clusters).
- **Halls ($H$):** Episodic traces (Bottom-level verbatim tokens).

## 3. Mathematical Unification
- **Differentiable Navigation:** Instead of a simple `retrieve()` call, the model uses a hierarchical attention mechanism to "query" the Palace.
- **Hamiltonian Coupling:** The Palace acts as a **Memory Potential ($V_{mem}$)** within the Hamiltonian. The world state $z$ is attracted to relevant "Halls" in the Palace, ensuring that predictions are grounded in past experience.
- **Update Rule:** Uses a differentiable GRU-style gate to consolidate new sensory input into the appropriate "Room" of the Palace.

## 4. 2.1B Scale Calibration
- **Embed Dim:** 2048
- **Depth:** 20 layers
- **Heads:** 32
- **LMP Subspaces:** 512-dim per wing.

## 5. Verification
- **Gate 1:** Successful backprop through the hierarchical attention layers.
- **Gate 2:** Verification that memory recall influences the ODE vector field during `odeint`.
