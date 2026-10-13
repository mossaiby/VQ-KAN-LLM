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

     $$\phi_k(x) = \operatorname{ReLU}(x - g_k)^2, \quad g_k \in [-1, 1]$$


   - The layer output combines a linear residual with the contracted basis weights:

     $$y_o = \sum_i W^{\text{base}}_{oi} x_i + \text{scale}_o \sum_{i,k} W^{\text{kan}}_{oik}\,\phi_k(x_i)$$


2. **Hypersphere Vector Quantization (`VectorQuantizer`)**
   - Both latent states $z$ and codebook entries $e_k$ are constrained via $L_2$ normalization onto a unit hypersphere ($S^{d-1}$).
   - Squared Euclidean distance is calculated as:

     $$D(z, e_k) = \lVert z - e_k \rVert_2^2 = 2 - 2(z \cdot e_k)$$


   - Gradients pass through the discrete `argmin` bottleneck using a Straight-Through Estimator (STE).
   - **Commitment loss weighting:** balanced at $\beta = 0.05$ to keep latent trajectories adaptive during early training.

3. **Domain-Matched Custom BPE Tokenizer**
   - Trained directly on a slice of the target corpus (`roneneldan/TinyStories`) using byte-level BPE.
   - The vocabulary is constrained to **2,048 tokens**, matching the size of the VQ codebook 1:1 and eliminating out-of-bounds or modulo-wrapped indexing.
   - Employs `ByteLevelDecoder` to restore whitespace and avoid visible byte-level artifacts such as `Ġ`.

4. **Aligned Loss Function & Cosine Classification Head**
   - The total objective combines cross-entropy in cosine-logit space with a latent commitment penalty:

   $$\mathcal{L} = \operatorname{CE}\!\left(16 \, \hat{z} \hat{E}^{\mathsf{T}}, y\right) + \mathcal{L}_{\mathrm{VQ}}$$


   where \(\hat{z} = z / \lVert z \rVert_2\) and each row of \(\hat{E}\) is an L2-normalized codebook vector.

   - Training and generation use the same logit space, preventing alignment drift between optimization and sampling.

---

## Project Structure

```text
VQ-KAN_LLM/
├── config.json                     # Central configuration for model, data, and training
├── kan_vq_transformer_scaled.py    # End-to-end training and checkpointing pipeline
├── generate.py                     # Standalone inference and interactive REPL
├── requirements.txt                # Core runtime dependencies
└── kan_vq_scaled_checkpoints/      # Checkpoint storage (generated during training)
    ├── tokenizer.json              # Serialized custom BPE tokenizer
    ├── best_model.pt               # State dict at lowest training loss
    ├── last_checkpoint.pt          # Full model, optimizer state, and training logs
    └── metrics_over_iterations.png # Training loss and batch code-utilization plot
```

---

## Configuration (`config.json`)

All architectural dimensions, dataset paths, training parameters, and inference settings are controlled via `config.json`.

```json
{
  "model": {
    "vocab_size": 2048,
    "embed_dim": 256,
    "num_layers": 4,
    "nhead": 8,
    "max_seq_len": 128,
    "dropout": 0.1,
    "grid_size": 5,
    "commitment_cost": 0.05
  },
  "data": {
    "dataset_name": "roneneldan/TinyStories",
    "dataset_split": "train[:25%]",
    "local_data_file": null
  },
  "training": {
    "batch_size": 32,
    "max_steps": 10000,
    "warmup_steps": 500,
    "max_lr": 4e-4,
    "min_lr": 4e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "log_interval": 50
  },
  "inference": {
    "max_new_tokens": 80,
    "temperature": 0.7,
    "top_k": 30,
    "top_p": 0.9,
    "repetition_penalty": 1.25
  },
  "paths": {
    "checkpoint_dir": "./kan_vq_scaled_checkpoints"
  }
}

```

### Parameter Reference

#### `model`

* **`vocab_size`**: Total number of subword tokens in the custom BPE tokenizer and discrete vectors in the VQ codebook.
* **`embed_dim`**: Latent feature dimension for token embeddings, attention representations, and the VQ bottleneck.
* **`num_layers`**: Number of stacked Transformer blocks containing causal self-attention and ReLU-KAN MLPs.
* **`nhead`**: Number of attention heads in the MultiheadAttention layers (`embed_dim // nhead` per head).
* **`max_seq_len`**: Context window length (maximum sequence length processed autoregressively).
* **`dropout`**: Dropout probability applied after attention projections and within the KAN MLP blocks.
* **`grid_size`**: Number of fixed knot points $g_k$ linearly spaced across $[-1, 1]$ for the squared-ReLU basis evaluation.
* **`commitment_cost`**: Scaling factor $\beta$ for the encoder latent MSE commitment loss against quantized vectors.

#### `data`

* **`dataset_name`**: Hugging Face Hub repository identifier for the target text dataset.
* **`dataset_split`**: Split slice to stream or download (e.g., `"train[:25%]"`).
* **`local_data_file`**: Optional path to a local `.txt` file (one sample per line). If specified, overrides the Hugging Face dataset download.

#### `training`

* **`batch_size`**: Number of independent token sequences sampled per forward-backward pass.
* **`max_steps`**: Total number of optimization steps to execute.
* **`warmup_steps`**: Steps allocated to linearly ramping the learning rate from $0$ to `max_lr`.
* **`max_lr`**: Peak learning rate achieved at the end of the warmup schedule.
* **`min_lr`**: Terminal learning rate at the end of cosine annealing decay.
* **`weight_decay`**: Weight decay coefficient used by the AdamW optimizer.
* **`grad_clip`**: Maximum gradient norm threshold for gradient clipping (`0.0` to disable).
* **`log_interval`**: Step frequency for logging training metrics to the console and tracking history for plotting.

#### `inference`

* **`max_new_tokens`**: Maximum number of autoregressive tokens to generate per prompt.
* **`temperature`**: Softmax logit scale controlling sampling randomness (`< 1.0` is sharper, `> 1.0` is flatter).
* **`top_k`**: Retains only the $k$ highest-probability tokens prior to multinomial sampling (`null` to disable).
* **`top_p`**: Nucleus threshold retaining the smallest set of tokens with cumulative probability $\ge p$ (`null` to disable).
* **`repetition_penalty`**: Divisive penalty applied to the logits of previously sampled tokens in the active context window.

#### `paths`

* **`checkpoint_dir`**: Directory where `tokenizer.json`, `best_model.pt`, `last_checkpoint.pt`, and `metrics_over_iterations.png` are stored.

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
python kan_vq_transformer_scaled.py --config config.json

```

### What happens during execution

1. Reads hyperparameters and directory paths from `config.json`.
2. Downloads the configured slice of TinyStories or loads the specified local corpus.
3. Checks for `tokenizer.json` in the checkpoint directory. If missing, trains a custom BPE tokenizer matching `vocab_size` and saves it to disk.
4. Initializes the KAN-VQ Transformer using `grid_size`, `num_layers`, `embed_dim`, and other settings from the configuration.
5. Trains for `max_steps` with AdamW and a warmup plus cosine-annealing learning-rate schedule.
6. Tracks loss and batch-level codebook usage.
7. Saves `best_model.pt` when a new minimum loss is reached.
8. Generates a training diagnostic plot at `metrics_over_iterations.png` within the checkpoint directory.

---

## Inference

Run interactive generation using the saved checkpoint and configuration settings:

```bash
python generate.py --config config.json

```

Generation parameters defined under the `inference` block in `config.json` (such as `temperature`, `top_k`, `top_p`, and `repetition_penalty`) are applied automatically.

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

* **Best recorded loss:** 2.1630. This is lower than the loss shown at step 10,000 (2.4756), so the best value occurred at an earlier checkpoint if these figures use the same loss metric.

* **Codebook utilization:** The listed checkpoints show 251–572 active codes from steps 500–10,000. The initial step reports 601 active codes. These snapshots suggest broad code usage, though they do not by themselves establish that no codebook entries become inactive over the full training run.