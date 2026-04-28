"""
Decoder-only Transformer model.
Implementation includes:
- Rotary positional embeddings.
- PyTorch optimized kernels for attention.
- LayerNorm applied before the attention and MLP blocks.
- No bias in the query, key, and value projections.
- GeLU activation function.
- Dropout applied after the token embedding.
"""

import os
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary positional embeddings to tensor x, assuming cos and sin
    tensors are already correctly sized and sliced, if needed.
    """
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Build cos/sin cache up to seq_len.
        """
        self._seq_len_cached = seq_len

        # Use float32 for better numerical precision in frequency computation
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))

        # Shape: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Convert to target dtype and add batch/head dimensions
        # No casting to (bfloat16) dtype here causes memory usage to explode
        self._cos_cached = emb.cos().to(dtype)[None, None, :, :]
        self._sin_cached = emb.sin().to(dtype)[None, None, :, :]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_pos_start: int = 0,
        k_pos_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor [batch, n_heads, seq_len_q, head_dim]
            k: Key tensor [batch, n_heads, seq_len_k, head_dim]
            q_pos_start: Starting position for queries (for KV caching)
            k_pos_start: Starting position for keys (usually 0 for cross-attn)

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]

        # Calculate maximum position needed
        max_pos = max(q_pos_start + seq_len_q, k_pos_start + seq_len_k)

        # Extend cache if needed (or initialize on first call)
        if (
            self._cos_cached is None
            or max_pos > self._seq_len_cached
            or self._cos_cached.device != q.device
            or self._cos_cached.dtype != q.dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            # Allocate cache with additional length to reduce future reallocations
            cache_len = max(max_pos, self._seq_len_cached * 2)
            self._build_cache(cache_len, q.device, q.dtype)
        # Extract cos/sin slices for q and k
        # Using narrow (view operation) instead of slicing for better performance
        q_cos = self._cos_cached.narrow(2, q_pos_start, seq_len_q)
        q_sin = self._sin_cached.narrow(2, q_pos_start, seq_len_q)

        if q_pos_start == k_pos_start and seq_len_q == seq_len_k:
            k_cos, k_sin = q_cos, q_sin
        else:
            k_cos = self._cos_cached.narrow(2, k_pos_start, seq_len_k)
            k_sin = self._sin_cached.narrow(2, k_pos_start, seq_len_k)

        # Apply rotary embeddings
        return (
            apply_rotary_pos_emb(q, q_cos, q_sin),
            apply_rotary_pos_emb(k, k_cos, k_sin),
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

        B, Tq, C = x.size()
        # Dropot must be set manually for F.scaled_dot_product_attention
        dropout_p = self.dropout_p if self.training else 0.0
        if self.is_causal and mask is not None:
            mask = None

        # Project queries
        q = self.q_proj(x).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        q_pos_start = past_kv[0].shape[2] if past_kv is not None else 0

        # Determine key/value source
        if kv is None:  # self-attention
            k = self.k_proj(x).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
            # Both q and k have same position offset in self-attention (with no grouping)
            k_pos_start = q_pos_start

        else:  # cross-attention
            k = self.k_proj(kv).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(kv).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)
            # q uses decoder position, k uses encoder position (starts at 0)
            k_pos_start = 0

        # Apply rotary embeddings
        q, k = self.rotary(q, k, q_pos_start=q_pos_start, k_pos_start=k_pos_start)

        # Append cached keys/values if present
        if past_kv is not None:  # TODO: check for cross-attention case
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # concatenated k is already rotated
            v = torch.cat([past_v, v], dim=2)

        Tk = k.size(2)

        # Compute attention output
        # The following is adapted from https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
        # to handle KV caching with optimized attention
        if past_kv is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=self.is_causal, dropout_p=dropout_p
            )
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            output = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout_p)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:  # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, attn_mask=attn_mask, dropout_p=dropout_p
            )

        # Merge heads
        output = (
            output.transpose(1, 2).contiguous().view(B, Tq, C)
        )  # faster than transpose + reshape
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
        use_cache: bool = False,  # Whether to use KV caching during generation
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        """
        Generate new tokens autoregressively.
        If eos_id is provided, stops generation for sequences that produce EOS.
        """
        B = x.shape[0]
        past_kv = None

        # Track finished sequences only if eos_id is given
        mask_completed = (
            torch.zeros(B, dtype=torch.bool, device=x.device) if eos_id is not None else None
        )

        for i in range(max_new_tokens):

            if i == 0 or not use_cache:
                output = self.forward(x, past_kv=past_kv, use_cache=use_cache)
            else:
                output = self.forward(x[:, -1:], past_kv=past_kv, use_cache=use_cache)

            past_kv = output.get("past_kv")
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


def load_model(ckpt_path: str, config_file: str | None = None) -> GPTTransformer:

    # load model config
    if config_file is None:
        config_file = os.path.join(os.path.dirname(ckpt_path), "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found at {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)

    model = GPTTransformer(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout_p=config.get("dropout_p", 0.1),
        mlp_ratio=config.get("mlp_ratio", 4),
        pad_id=config.get("pad_id", 0),
    )

    # load model from pt file inside the checkpoint folder
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    return model
