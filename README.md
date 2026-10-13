# KAN-VQ Transformer: Autoregressive Language Modeling via ReLU-KAN & Hypersphere Vector Quantization

An experimental autoregressive language model built from scratch in PyTorch[cite: 1]. The architecture replaces traditional feed-forward networks (MLPs) with Kolmogorov-Arnold Network layers based on squared-ReLU basis functions (ReLU-KAN), pairs them with a Vector Quantization (VQ) codebook, and optimizes over a custom Byte-Pair Encoding (BPE) vocabulary using cosine-scaled classification on a unit hypersphere[cite: 1].

---

## Architectural Overview


```
Input Tokens: x_{1:T}
    │
    ▼
Embedding (Token + Positional) ──► Latent dim: 256
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Transformer Block (x4)                                  │
│   ├── LayerNorm ──► Causal Multihead Attention (8 heads)│
│   │     └── Residual Add + Dropout                      │
│   └── LayerNorm ──► ReLUKANMLP (Expansion: 4x)          │
│           ├── ReLUKANLinear (256 -> 1024)               │
│           │     ├── Base Linear shortcut                │
│           │     └── Squared-ReLU Grid Contraction       │
│           ├── ReLU Activation                           │
│           └── ReLUKANLinear (1024 -> 256)               │
│                 └── Residual Add + Dropout              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Vector Quantization & Projection Head
├── L2 Hypersphere Normalization: z / ||z||_2
├── Euclidean Distance to Codebook: d = 2 - 2 * cos(z, E)
└── Cosine-Scaled Classification: Logits = 16.0 * (z · E^T)
```

### Key Components

1. **ReLU-KAN Feed-Forward Layers (`ReLUKANLinear`)**
   * Unlike standard KAN formulations that rely on computationally expensive B-splines and spline coefficient matrices, this model implements ReLU-KAN[cite: 1].
   * Activation functions on edges are evaluated via parametric squared-ReLU functions relative to fixed knot points:  
     $$\phi_k(x) = \text{ReLU}(x - g_k)^2, \quad g_k \in [-1, 1]$$
   * The layer output combines a linear residual with the contracted basis weights:  
     $$y_o = \sum_i W^{\text{base}}_{oi} x_i + \text{scale}_o \cdot \sum_{i, k} W^{\text{kan}}_{oik} \cdot \phi_k(x_i)$$

2. **Hypersphere Vector Quantization (`VectorQuantizer`)**
   * Both latent states $z$ and codebook entries $e_k$ are constrained via $L_2$ normalization onto a unit hypersphere ($S^{d-1}$)[cite: 1].
   * Spatial Euclidean distance is calculated as:  
     $$D(z, e_k) = \Vert{}z - e_k\Vert{}_2^2 = 2 - 2 \cdot (z \cdot e_k)$$
[cite: 1]
   * Gradients pass through the discrete argmin bottleneck using a Straight-Through Estimator (STE)[cite: 1].
   * **Commitment Loss Weighting:** Balanced at $\beta = 0.05$ to keep latent trajectories adaptive during early training[cite: 1].

3. **Domain-Matched Custom BPE Tokenizer**
   * Trained directly on a slice of the target corpus (`roneneldan/TinyStories`) using Byte-Level BPE[cite: 1].
   * Vocabulary is strictly constrained to **2,048 tokens**, matching the size of the VQ codebook 1:1 and eliminating out-of-bounds or modulo-wrapped indexing.
   * Employs `ByteLevelDecoder` to ensure clean whitespace restoration and eliminate byte artifacts (`Ġ`)[cite: 1].

4. **Aligned Loss Function & Cosine Classification Head**
   * Total objective combines spatial cross-entropy and latent commitment penalty:  
     $$\mathcal{L} = \text{CrossEntropy}\left(16.0 \cdot \frac{z}{\Vert{}z\Vert{}_2} \left(\frac{E}{\Vert{}E\Vert{}_2}\right)^T, y\right) + \mathcal{L}_{\text{VQ}}$$
[cite: 1]
   * Training and generation are identical in logit space, preventing alignment drift between optimization and sampling[cite: 1].

---

## Project Structure


```
VQ-KAN_LLM/
├── kan_vq_transformer_scaled_up.py   # End-to-end training and checkpointing pipeline
├── generate.py                       # Standalone inference and interactive REPL
├── requirements.txt                  # Core runtime dependencies
└── kan_vq_scaled_checkpoints/        # Checkpoint storage (generated during training)
   ├── tokenizer.json                 # Serialized custom BPE tokenizer
   ├── best_model.pt                  # State dict at lowest training loss
   ├── last_checkpoint.pt             # Full model, optimizer state, and training logs
   └── metrics_over_iterations.png    # Training loss and batch code utilization plot
```

---

## Requirements

* Python 3.10+[cite: 1]
* CUDA-capable GPU recommended[cite: 1]

Install dependencies[cite: 1]:

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

### What happens during execution:

1. Downloads a 25% slice of the TinyStories dataset (~529,930 stories).
2. Checks for `kan_vq_scaled_checkpoints/tokenizer.json`. If missing, trains a fresh 2,048-token BPE tokenizer and serializes it to disk.
3. Initializes the 4-layer, 14.7M parameter KAN-VQ Transformer.
4. Trains for 10,000 steps with AdamW and a Warmup + Cosine Annealing learning rate schedule ($\text{max\_lr} = 4 \times 10^{-4}$, $\text{min\_lr} = 4 \times 10^{-5}$, batch size 32, sequence length 128).
5. Monitors loss and active batch code utilization.
6. Automatically saves `best_model.pt` whenever a new loss minimum is achieved.
7. Generates a training diagnostic plot to `kan_vq_scaled_checkpoints/metrics_over_iterations.png`.



---

## Inference

Run interactive generation using the saved checkpoint:

```bash
python generate.py

```

### Generation Parameters Supported:

* **Temperature** (default: `0.7`): Controls sharpness of the softmax probability distribution.


* **Top-K** (default: `30`): Restricts selection to the $k$ most likely next tokens.
* **Top-P / Nucleus** (default: `0.9`): Restricts selection to the smallest token set whose cumulative probability reaches $p$.


* **Repetition Penalty** (default: `1.25`): Scales down logits of previously generated tokens to prevent autoregressive degeneration loops.



---

## Training Convergence Summary

| Step | Learning Rate | Total Loss ($\text{Cosine CE} + \mathcal{L}_{\text{VQ}}$) | Active Codes in Batch (out of 2,048) |
| --- | --- | --- | --- |
| **1** | $1.60 \times 10^{-6}$ | 8.0757 | 601 |
| **500** | $4.00 \times 10^{-4}$ | 3.7652 | 251 |
| **1000** | $3.98 \times 10^{-4}$ | 3.4440 | 364 |
| **2500** | $3.62 \times 10^{-4}$ | 2.9082 | 467 |
| **5000** | $2.35 \times 10^{-4}$ | 2.5520 | 514 |
| **7500** | $9.81 \times 10^{-5}$ | 2.3676 | 555 |
| **10000** | $4.00 \times 10^{-5}$ | 2.4756 | 572 |

* **Convergence Point:** Lowest loss reaches **2.1630**.
* **Codebook Stability:** Active tokens per batch steadily expand from ~250 up to ~570 without dead-code collapse or loss spikes.
```