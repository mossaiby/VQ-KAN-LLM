import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# 1. ReLU-KAN Linear Layer Implementation
# ==========================================
class ReLUKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        self.base_linear = nn.Linear(in_features, out_features)
        self.kan_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.scale_factor = nn.Parameter(torch.ones(out_features))
        
        nn.init.kaiming_uniform_(self.base_linear.weight, nonlinearity='relu')
        nn.init.normal_(self.kan_weight, mean=0.0, std=1.0 / (in_features ** 0.5))
        
        grid = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer('grid', grid)

    def forward(self, x):
        base_out = self.base_linear(x)
        x_expanded = x.unsqueeze(-1)
        h = F.relu(x_expanded - self.grid) ** 2
        kan_out = torch.einsum('...if,oif->...o', h, self.kan_weight)
        return base_out + kan_out * self.scale_factor


# ==========================================
# 2. Transformer MLP Block using ReLU-KAN
# ==========================================
class ReLUKANMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.kan1 = ReLUKANLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.kan2 = ReLUKANLinear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.kan1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.kan2(x)
        return x


# ==========================================
# 3. Vector Quantizer (Unit-Sphere VQ)
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.05):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, z):
        flat_z = F.normalize(z.reshape(-1, self.embedding_dim), p=2, dim=-1)
        norm_emb = F.normalize(self.embedding.weight, p=2, dim=-1)
        
        distances = 2.0 - 2.0 * torch.matmul(flat_z, norm_emb.t())
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = norm_emb[encoding_indices].view(z.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), flat_z.view(z.shape))
        q_latent_loss = F.mse_loss(quantized, flat_z.view(z.shape).detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = z + (quantized - z).detach()
        reshaped_distances = distances.view(z.shape[0], z.shape[1], self.num_embeddings)
        return quantized, vq_loss, encoding_indices.view(z.shape[0], z.shape[1]), reshaped_distances


# ==========================================
# 4. Scaled KAN-VQ Transformer
# ==========================================
class KAN_VQTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, nhead=8, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(embed_dim, nhead, batch_first=True, dropout=dropout),
                'mlp': ReLUKANMLP(embed_dim, embed_dim * 4, dropout=dropout),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        
        self.vq_layer = VectorQuantizer(num_embeddings=vocab_size, embedding_dim=embed_dim)

    def forward(self, idx):
        b, s = idx.shape
        device = idx.device
        positions = torch.arange(0, s, device=device).unsqueeze(0).expand(b, s)
        x = self.token_embedding(idx) + self.pos_embedding(positions[:, :s])
        
        mask = nn.Transformer.generate_square_subsequent_mask(s).to(device)
        
        for layer in self.layers:
            norm_x = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](norm_x, norm_x, norm_x, attn_mask=mask, is_causal=True)
            x = x + layer['dropout'](attn_out)
            
            norm_x = layer['norm2'](x)
            mlp_out = layer['mlp'](norm_x)
            x = x + layer['dropout'](mlp_out)
        
        quantized_out, vq_loss, code_indices, distances = self.vq_layer(x)
        return quantized_out, vq_loss, code_indices, distances, x


# ==========================================
# 5. Learning Rate Scheduler (Warmup + Cosine)
# ==========================================
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ==========================================
# 6. Metric Plotting Helper
# ==========================================
def save_metrics_plot(history_steps, history_loss, history_codes, vocab_size, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(history_steps, history_loss, color="#1f77b4", linewidth=1.8, label="Total Loss")
    ax1.set_ylabel("Loss (Cosine CE + VQ)", fontsize=11)
    ax1.set_title("Scaled KAN-VQ Training Dynamics", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(history_steps, history_codes, color="#2ca02c", linewidth=1.8, label="Unique Codes in Batch")
    ax2.axhline(y=vocab_size, color="#d62728", linestyle=":", linewidth=1.5, label=f"Total Vocab ({vocab_size})")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Unique Tokens in Batch", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ==========================================
# 7. Training & Setup
# ==========================================
if __name__ == "__main__":
    # Scaled Hyperparameters
    embed_dim = 256
    num_layers = 4
    nhead = 8
    batch_size = 32
    seq_len = 128
    vocab_size = 2048
    max_steps = 10000
    warmup_steps = 500
    max_lr = 4e-4
    min_lr = 4e-5
    log_interval = 50

    save_dir = "./kan_vq_scaled_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_file = os.path.join(save_dir, "tokenizer.json")

    print("Loading TinyStories dataset (25% slice)...")
    try:
        raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:25%]")
        dataset_texts = raw_dataset.to_dict()['text']
        print(f"Loaded {len(dataset_texts):,} stories.")
    except Exception:
        dataset_texts = [
            "Once upon a time, there was a little cat named Lily. Lily loved to play with a red ball."
        ] * 100

    if os.path.exists(tokenizer_file):
        print(f"Loading cached tokenizer from: {tokenizer_file}")
        raw_tokenizer = Tokenizer.from_file(tokenizer_file)
    else:
        print(f"Training custom BPE tokenizer with vocab size {vocab_size}...")
        raw_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        raw_tokenizer.pre_tokenizer = ByteLevel()
        raw_tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
        )
        raw_tokenizer.train_from_iterator(dataset_texts, trainer)
        raw_tokenizer.save(tokenizer_file)
        print(f"Tokenizer saved to: {tokenizer_file}")

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    tokenizer.pad_token = "<pad>"

    def get_batch():
        batch_texts = [random.choice(dataset_texts) for _ in range(batch_size)]
        encoded_batches = []
        for text in batch_texts:
            ids = tokenizer.encode(text)
            if len(ids) < seq_len + 1:
                ids = ids + [tokenizer.pad_token_id] * (seq_len + 1 - len(ids))
            else:
                ids = ids[:seq_len + 1]
            encoded_batches.append(ids)
            
        data_tensor = torch.tensor(encoded_batches, dtype=torch.long)
        return data_tensor[:, :-1], data_tensor[:, 1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Scaled KAN-VQ Transformer on device: {device}")

    model = KAN_VQTransformer(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        num_layers=num_layers, 
        nhead=nhead, 
        max_seq_len=seq_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

    history_steps = []
    history_loss = []
    history_codes = []

    print(f"\nStarting training loop for {max_steps} steps...\n")
    print(f"{'Step':<7} | {'LR':<9} | {'Loss':<10} | {'Batch Codes':<12}")
    print("-" * 46)

    best_loss = float('inf')

    for step in range(1, max_steps + 1):
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        x_batch, y_batch = get_batch()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        quantized_out, vq_loss, code_indices, distances, raw_out = model(x_batch)
        
        flat_out = F.normalize(raw_out.reshape(-1, embed_dim), p=2, dim=-1)
        norm_weights = F.normalize(model.vq_layer.embedding.weight, p=2, dim=-1)
        logits = torch.matmul(flat_out, norm_weights.t()) * 16.0
        
        ce_loss = F.cross_entropy(logits, y_batch.reshape(-1))
        total_loss = ce_loss + vq_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = total_loss.item()
        unique_codes = torch.unique(code_indices).numel()

        if step == 1 or step % log_interval == 0:
            history_steps.append(step)
            history_loss.append(current_loss)
            history_codes.append(unique_codes)

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        if step == 1 or step % 500 == 0:
            print(f"{step:<7} | {lr:<9.2e} | {current_loss:<10.4f} | {unique_codes:<4} in batch")

    # Save final checkpoint
    torch.save({
        'step': max_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'nhead': nhead,
        'seq_len': seq_len
    }, os.path.join(save_dir, "last_checkpoint.pt"))

    print(f"\nTraining complete. Lowest loss: {best_loss:.4f}")
    plot_path = os.path.join(save_dir, "metrics_over_iterations.png")
    save_metrics_plot(history_steps, history_loss, history_codes, vocab_size, plot_path)
    print(f"Metrics plot saved to: {plot_path}")

    # ==========================================
    # 8. Generation Test
    # ==========================================
    print("\n" + "="*45)
    print(" SCALED KAN-VQ GENERATION TEST")
    print("="*45)

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    model.eval()

    @torch.no_grad()
    def generate(prompt_str, max_new_tokens=60, temperature=0.7, top_k=30, top_p=0.9):
        input_ids = tokenizer.encode(prompt_str)
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -seq_len:]
            _, _, _, _, raw_out = model(idx_cond)
            
            flat_out = F.normalize(raw_out[:, -1:, :], p=2, dim=-1)
            norm_weights = F.normalize(model.vq_layer.embedding.weight, p=2, dim=-1)
            next_token_logits = (torch.matmul(flat_out, norm_weights.t()) * 16.0).squeeze(1)

            # Repetition penalty
            for token_id in set(idx[0].tolist()):
                next_token_logits[0, token_id] /= 1.25

            next_token_logits = next_token_logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)

    prompt = "Once upon a time"
    print(f"Prompt: '{prompt}'\n")
    print(f"Generated Result:\n{generate(prompt)}")