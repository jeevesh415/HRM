# Hierarchical Reasoning Model - V-JEPA Integration (AGI Scale)

This repository hosts the advanced integration of the Hierarchical Reasoning Model (HRM) with the Video Joint-Embedding Predictive Architecture (V-JEPA), scaled to a **10 Billion parameter** architecture for deep physical world understanding.

## Core Vision
Transitioning from discrete puzzle-solving to continuous-time, latent-space reasoning. The model is designed to learn **intuitive physics** (depth, shadows, object permanence, continuity) autonomously from raw video data, achieving a human-like understanding of the physical world.

## Key Architectural Components

### 1. Vision Encoder (The Eyes)
*   **3D Patch Embedding**: Processes video clips as spatio-temporal volumes.
*   **3D-RoPE**: 3D Rotary Positional Embeddings that natively encode Time, Height, and Width coordinates.
*   **10B Scale ViT**: A massive Vision Transformer designed to capture high-density visual information.

### 2. The HRM-ODE Predictor (The Brain)
*   **Neural ODEs**: Continuous-time reasoning cycles that model the differential equations of physical dynamics ($dz/dt$).
*   **Holographic Memory**: Vector Symbolic Architecture (VSA) based memory that binds and stores complex physical experiences into dense, high-dimensional holographic states.
*   **Hierarchical Latent Reasoning**: Two-level latent states ($z_H$ and $z_L$) for global physical planning and fine-grained patch refinement.

### 3. Latent Planning (The Imagination)
*   **Latent MCTS**: Monte Carlo Tree Search operating entirely in latent space, allowing the model to "imagine" and evaluate thousands of future physical outcomes.
*   **Action Conditioning**: Future states predicted conditioned on specific physical actions/interventions.

### 4. Advanced Training Engine
*   **VICReg Objective**: Variance-Covariance regularization to prevent representation collapse and ensure a diverse latent manifold.
*   **3D Block Masking**: Spatio-temporal masking that forces the model to infer large missing segments of the world across space and time.
*   **Distributed Scaling**: Ready for FSDP and DeepSpeed multi-node training.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Adjust the 10B parameter specs in `config/vjepa_10b.yaml`.

### Training
```bash
python vjepa_train.py
```

## Future Multimodal Grounding
The architecture is designed to be modality-agnostic, with hooks ready for future integration of **Audio** and **Tactile (Proprioceptive)** data to achieve true multisensory physical grounding.

---
*This project is dedicated to pushing the boundaries of artificial general intelligence through the lens of hierarchical physical reasoning.*
