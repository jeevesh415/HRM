# Hierarchical Reasoning Model - V-JEPA Integration (AGI Scale)

This repository hosts the advanced integration of the Hierarchical Reasoning Model (HRM) with the Video Joint-Embedding Predictive Architecture (V-JEPA), scaled to a **10 Billion parameter** architecture for deep physical world understanding.

## Core Vision
Transitioning from discrete puzzle-solving to continuous-time, latent-space reasoning. The model is designed to learn **intuitive physics** (depth, shadows, object permanence, continuity) autonomously from raw video data, achieving a human-like understanding of the physical world.

## Key Architectural Pillars

### 1. Vision Encoder (The Eyes)
*   **3D Patch Embedding**: Processes video clips as spatio-temporal volumes.
*   **3D-RoPE**: 3D Rotary Positional Embeddings that natively encode Time, Height, and Width coordinates.
*   **10B Scale ViT**: A massive Vision Transformer designed to capture high-density visual information.

### 2. Physical Relativity (Lie Group Equivariance)
*   **Stiefel Manifold Projections**: Implements $O(D)$ complexity equivariant transformations using Cayley transforms, ensuring physical laws are relative across 10B parameter manifolds.

### 3. Continuous-Time Brain (Hamiltonian Neural ODEs & Predictive Coding)
*   **Symplectic Physics Engine (HNN)**: Uses **Hamiltonian Neural Networks** to compute the continuous-time dynamics ($dq/dt$, $dp/dt$). This guarantees absolute energy conservation and strict adherence to classical mechanics within the latent imagination space.
*   **Adjoint Neural ODEs**: Uses the **Neural ODE Adjoint Method** (`torchdiffeq`) for constant-memory backpropagation, enabling infinite-depth continuous reasoning at 10B scale.
*   **Top-Down Predictive Coding**: A hierarchical "handshake" where High-Level planning ($z_H$) suppresses error signals from Low-Level sensors ($z_L$), mimicking the human visual cortex.
*   **Holographic Memory**: Vector Symbolic Architecture (VSA) based memory that binds and stores complex physical experiences into dense, high-dimensional holographic states.

### 4. Light & Shadow Intuition (Neural Radiance Latents)
*   **Volumetric Ray-Marching**: Treats the latent space as a **Differentiable Continuous Radiance Field (NeRF)**. The model "traces" light and reflections through its imagined 3D manifold.

### 5. Latent Planning (The Imagination)
*   **Latent MCTS**: Monte Carlo Tree Search operating entirely in latent space, allowing the model to "imagine" and evaluate thousands of future physical outcomes.
*   **Action Conditioning**: Future states predicted conditioned on specific physical actions/interventions.

### 6. Advanced Training Engine
*   **VICReg Objective**: Variance-Covariance regularization to prevent representation collapse.
*   **3D Block Masking**: Spatio-temporal masking that forces the model to infer large missing segments of the world.

## Getting Started

### Configuration
Adjust the 10B parameter specs in `config/vjepa_10b.yaml`.

### Training
```bash
python vjepa_train.py
```

## Future Multimodal Grounding
The architecture is designed to be modality-agnostic, with hooks ready for future integration of **Audio** and **Tactile (Proprioceptive)** data.

---
*This project is dedicated to pushing the boundaries of artificial general intelligence through the lens of hierarchical physical reasoning.*
