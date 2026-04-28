"""
Optimized version of Decoder-only Transformer model.
Decoder-only architecture notes:
- Rotary positional embeddings are used instead of absolute positional embeddings.
- PyTorch optimized kernels for dot-product attention.
- Layer normalization is applied before the attention and MLP blocks.
- No bias in the query, key, and value projections.
- GeLU activation function is used in the MLP.
- Dropout is applied after the token embedding.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_dimension: int = -2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        # or if we switch from inference mode to training
        if (
            seq_len > self._seq_len_cached
            or self._cos_cached.device != x.device
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout_p: float = 0.1, is_causal: bool = True):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)  # No bias in q, k, v projections
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout_p = dropout_p
        self.is_causal = is_causal

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if past_kv is not None or use_cache is True:
            raise NotImplementedError("KV caching is not implemented")
        
        B, T, C = x.size()
        # Dropot must be set manually for F.scaled_dot_product_attention
        dropout_p = self.dropout_p if self.training else 0.0  
        if self.is_causal and mask is not None:
            mask = None

        # Project queries
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Determine key/value source
        if kv is None:  # self-attention
            k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        else:  # cross-attention
            k = self.k_proj(kv).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(kv).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q, k = self.rotary(q, k)  # BUG: gives wrong results if past_kv is not None

        # Append cached keys/values if present
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Compute attention
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=self.is_causal
        )

        # Merge heads
        output = output.transpose(1, 2).reshape(B, T, C)
        output = self.o_proj(output)

        # Return cache if needed
        new_kv = (k, v) if use_cache else None
        return output, new_kv


class MLP(nn.Module):
    def __init__(self, embed_dim: int, dropout_p: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        upscale_dim = embed_dim * mlp_ratio
        self.up_proj = nn.Linear(embed_dim, upscale_dim)
        self.down_proj = nn.Linear(upscale_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.up_proj(x)))
        x = self.down_proj(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p, mlp_ratio, is_causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout_p, is_causal)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout_p, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        x_att, new_kv = self.attn(
            self.ln1(x), kv=kv, mask=mask, past_kv=past_kv, use_cache=use_cache
        )
        x = x + x_att
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class GPTTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout_p, mlp_ratio, pad_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, num_heads, dropout_p, mlp_ratio, is_causal=True)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        
        if past_kv is None:
            past_kv = [None] * self.num_layers

        # Embed
        x = self.token_emb(input_ids)

        # Transformer stack
        for i, layer in enumerate(self.layers):
            x, past_kv[i] = layer(x, past_kv=past_kv[i], use_cache=use_cache)

        # Decode
        x = self.ln(x)
        logits = self.unembed(x)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_id,
            )

        if not use_cache:
            past_kv = None  # convert list of None to single None

        return {"logits": logits, "loss": loss, "past_kv": past_kv}

    @torch.inference_mode()
    def generate(
        self,
        x: torch.LongTensor,
        max_new_tokens: int,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        """
        Generate new tokens autoregressively.
        If eos_id is provided, stops generation for sequences that produce EOS.
        """
        B = x.shape[0]

        # Track finished sequences only if eos_id is given
        mask_completed = (
            torch.zeros(B, dtype=torch.bool, device=x.device) if eos_id is not None else None
        )

        for _ in range(max_new_tokens):
            output = self.forward(x, past_kv=None, use_cache=False)
            logits = output["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, [-1]]] = -float("inf")

            # Mask sequences that are done
            if eos_id is not None and mask_completed.any():
                logits[mask_completed, :] = -float("inf")
                logits[mask_completed, self.pad_id] = 0.0

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)

            # Update done mask
            if eos_id is not None:
                mask_completed = mask_completed | (next_token.squeeze(-1) == eos_id)
                # Stop entirely if all sequences are done
                if mask_completed.all():
                    break

            # Append new token
            x = torch.cat((x, next_token), dim=1)

        return x


if __name__ == "__main__":
    model = GPTTransformer(
        vocab_size=32,
        embed_dim=64,
        num_heads=8,
        num_layers=8,
        mlp_ratio=4,
        dropout_p=0.1,
        pad_id=31,
    )
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input = torch.randint(0, 32, size=(128, 1)).to(device)
    output = model(input, targets=input)
    print(output)

    import time

    t0 = time.time()
    x = model.generate(input, max_new_tokens=30)
    print(input.shape, x.shape)
    print(f"Gen time: {time.time() - t0:.3f}")
