import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

# ==========================================
# 1. Model Definitions (Matching Scaled Architecture)
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

        grid = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer('grid', grid)

    def forward(self, x):
        base_out = self.base_linear(x)
        x_expanded = x.unsqueeze(-1)
        h = F.relu(x_expanded - self.grid) ** 2
        kan_out = torch.einsum('...if,oif->...o', h, self.kan_weight)
        return base_out + kan_out * self.scale_factor


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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.05):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, z):
        flat_z = F.normalize(z.reshape(-1, self.embedding_dim), p=2, dim=-1)
        norm_emb = F.normalize(self.embedding.weight, p=2, dim=-1)
        distances = 2.0 - 2.0 * torch.matmul(flat_z, norm_emb.t())
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = norm_emb[encoding_indices].view(z.shape)
        reshaped_distances = distances.view(z.shape[0], z.shape[1], self.num_embeddings)
        return quantized, None, encoding_indices.view(z.shape[0], z.shape[1]), reshaped_distances


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
# 2. Generation Engine
# ==========================================
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 30,
    top_p: float = 0.9,
    repetition_penalty: float = 1.25,
    device: str = "cuda"
):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.pad_token_id]

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    seq_len = model.pos_embedding.num_embeddings

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -seq_len:]
        _, _, _, _, raw_out = model(idx_cond)

        flat_out = F.normalize(raw_out[:, -1:, :], p=2, dim=-1)
        norm_weights = F.normalize(model.vq_layer.embedding.weight, p=2, dim=-1)
        next_token_logits = (torch.matmul(flat_out, norm_weights.t()) * 16.0).squeeze(1)

        # Repetition penalty
        for token_id in set(idx[0].tolist()):
            next_token_logits[0, token_id] /= repetition_penalty

        next_token_logits = next_token_logits / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

        if top_p is not None and 0.0 < top_p < 1.0:
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


# ==========================================
# 3. Interactive CLI Runner
# ==========================================
if __name__ == "__main__":
    checkpoints_dir = "./kan_vq_scaled_checkpoints"
    tokenizer_path = os.path.join(checkpoints_dir, "tokenizer.json")
    model_path = os.path.join(checkpoints_dir, "best_model.pt")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Missing tokenizer at {tokenizer_path}.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing weights at {model_path}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint onto {device}...")

    raw_tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    tokenizer.pad_token = "<pad>"

    # Scaled configuration matching checkpoint
    vocab_size = 2048
    embed_dim = 256
    num_layers = 4
    nhead = 8
    seq_len = 128

    model = KAN_VQTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        nhead=nhead,
        max_seq_len=seq_len
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nModel ready. Enter your prompt below (or type 'exit' to quit).\n" + "-" * 50)

    while True:
        try:
            prompt = input("Prompt >>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                break

            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80,
                temperature=0.7,
                top_k=30,
                top_p=0.9,
                repetition_penalty=1.25,
                device=device
            )

            print(f"\nGenerated:\n{output}\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\nExiting.")
            break