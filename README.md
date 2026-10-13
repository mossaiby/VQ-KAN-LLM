# KAN-VQ Transformer: Autoregressive Language Modeling via ReLU-KAN & Hypersphere Vector Quantization

An experimental autoregressive language model built from scratch in PyTorch. The architecture replaces traditional feed-forward networks (MLPs) with Kolmogorov–Arnold Network layers based on squared-ReLU basis functions (ReLU-KAN), pairs them with a vector-quantization (VQ) codebook, and optimizes over a custom byte-pair encoding (BPE) vocabulary using cosine-scaled classification on a unit hypersphere.

---

## Architectural Overview

```text
Input Tokens: x_{1:T}
    │
    ▼
Embedding (Token + Positional) ──► Latent dim: 256
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│ Transformer Block (×4)                                    │
│   ├── LayerNorm ──► Causal Multihead Attention (8 heads)  │
│   │       └── Residual Add + Dropout                      │
│   └── LayerNorm ──► ReLUKANMLP (Expansion: 4×)            │
│           ├── ReLUKANLinear (256 → 1024)                  │
│           │   ├── Base Linear shortcut                    │
│           │   └── Squared-ReLU Grid Contraction           │
│           ├── ReLU Activation                             │
│           └── ReLUKANLinear (1024 → 256)                  │
│                   └── Residual Add + Dropout              │
└───────────────────────────────────────────────────────────┘
    │
    ▼
Vector Quantization & Projection Head
    ├── L2 Hypersphere Normalization: z / ||z||₂
    ├── Euclidean Distance to Codebook: d = 2 − 2 · cos(z, E)
    └── Cosine-Scaled Classification: logits = 16.0 · (z · Eᵀ)
```

### Key Components

1. **ReLU-KAN Feed-Forward Layers (`ReLUKANLinear`)**
   - Unlike standard KAN formulations that rely on computationally expensive B-splines and spline coefficient matrices, this model implements ReLU-KAN.
   - Activation functions on edges are evaluated via parametric squared-ReLU functions relative to fixed knot points:

     $$
     \phi_k(x) = \operatorname{ReLU}(x - g_k)^2, \quad g_k \in [-1, 1]
     $$

   - The layer output combines a linear residual with the contracted basis weights:

     $$
     y_o = \sum_i W^{\text{base}}_{oi} x_i + \text{scale}_o \sum_{i,k} W^{\text{kan}}_{oik}\,\phi_k(x_i)
     $$

2. **Hypersphere Vector Quantization (`VectorQuantizer`)**
   - Both latent states $z$ and codebook entries $e_k$ are constrained via $L_2$ normalization onto a unit hypersphere ($S^{d-1}$).
   - Squared Euclidean distance is calculated as:

     $$
     D(z, e_k) = \lVert z - e_k \rVert_2^2 = 2 - 2(z \cdot e_k)
     $$

   - Gradients pass through the discrete `argmin` bottleneck using a Straight-Through Estimator (STE).
   - **Commitment loss weighting:** balanced at $\beta = 0.05$ to keep latent trajectories adaptive during early training.

3. **Domain-Matched Custom BPE Tokenizer**
   - Trained directly on a slice of the target corpus (`roneneldan/TinyStories`) using byte-level BPE.
   - The vocabulary is constrained to **2,048 tokens**, matching the size of the VQ codebook 1:1 and eliminating out-of-bounds or modulo-wrapped indexing.
   - Employs `ByteLevelDecoder` to restore whitespace and avoid visible byte-level artifacts such as `Ġ`.

4. **Aligned Loss Function & Cosine Classification Head**
   - The total objective combines cross-entropy in cosine-logit space with a latent commitment penalty:

   $$
   \mathcal{L} = \operatorname{CE}\!\left(16 \, \hat{z} \hat{E}^{\mathsf{T}}, y\right) + \mathcal{L}_{\mathrm{VQ}}
   $$

   where \(\hat{z} = z / \lVert z \rVert_2\) and each row of \(\hat{E}\) is an L2-normalized codebook vector.

   - Training and generation use the same logit space, preventing alignment drift between optimization and sampling.

---

## Project Structure

```text
VQ-KAN_LLM/
├── kan_vq_transformer_scaled_up.py  # End-to-end training and checkpointing pipeline
├── generate.py                      # Standalone inference and interactive REPL
├── requirements.txt                 # Core runtime dependencies
└── kan_vq_scaled_checkpoints/       # Checkpoint storage (generated during training)
    ├── tokenizer.json               # Serialized custom BPE tokenizer
    ├── best_model.pt                # State dict at lowest training loss
    ├── last_checkpoint.pt            # Full model, optimizer state, and training logs
    └── metrics_over_iterations.png  # Training loss and batch code-utilization plot
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
torch>=2.1.0
transformers>=4.38.0
datasets>=2.18.0
tokenizers>=0.15.0
matplotlib>=3.8.0
```

---

## Training

Run the scaled training pipeline:

```bash
python kan_vq_transformer_scaled_up.py
```

### What happens during execution

1. Downloads a 25% slice of the TinyStories dataset (approximately 529,930 stories).
2. Checks for `kan_vq_scaled_checkpoints/tokenizer.json`. If the file is missing, trains a fresh 2,048-token BPE tokenizer and serializes it to disk.
3. Initializes the 4-layer, 14.7-million-parameter KAN-VQ Transformer.
4. Trains for 10,000 steps with AdamW and a warmup plus cosine-annealing learning-rate schedule: `max_lr = 4e-4`, `min_lr = 4e-5`, batch size 32, and sequence length 128.
5. Monitors loss and active batch-code utilization.
6. Automatically saves `best_model.pt` whenever a new minimum loss is achieved.
7. Generates a training diagnostic plot at `kan_vq_scaled_checkpoints/metrics_over_iterations.png`.

---

## Inference

Run interactive generation using the saved checkpoint:

```bash
python generate.py
```

### Generation Parameters Supported

- **Temperature** (default: `0.7`): Controls the sharpness of the softmax probability distribution.
- **Top-K** (default: `30`): Restricts selection to the `k` most likely next tokens.
- **Top-P / nucleus sampling** (default: `0.9`): Restricts selection to the smallest token set whose cumulative probability reaches `p`.
- **Repetition penalty** (default: `1.25`): Scales down logits of previously generated tokens to reduce repetitive generation loops.

---

## Training Convergence Summary

| Step | Learning Rate | Total Loss (Cosine CE + $\mathcal{L}_{\mathrm{VQ}}$) | Active Codes in Batch (out of 2,048) |
| ---: | ---: | ---: | ---: |
| **1** | $1.60 \times 10^{-6}$ | 8.0757 | 601 |
| **500** | $4.00 \times 10^{-4}$ | 3.7652 | 251 |
| **1,000** | $3.98 \times 10^{-4}$ | 3.4440 | 364 |
| **2,500** | $3.62 \times 10^{-4}$ | 2.9082 | 467 |
| **5,000** | $2.35 \times 10^{-4}$ | 2.5520 | 514 |
| **7,500** | $9.81 \times 10^{-5}$ | 2.3676 | 555 |
| **10,000** | $4.00 \times 10^{-5}$ | 2.4756 | 572 |

- **Best recorded loss:** 2.1630. This is lower than the loss shown at step 10,000 (2.4756), so the best value occurred at an earlier checkpoint if these figures use the same loss metric.
- **Codebook utilization:** The listed checkpoints show 251–572 active codes from steps 500–10,000. The initial step reports 601 active codes. These snapshots suggest broad code usage, though they do not by themselves establish that no codebook entries become inactive over the full training run.
