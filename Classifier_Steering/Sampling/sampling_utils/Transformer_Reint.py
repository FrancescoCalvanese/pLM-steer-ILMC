import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple

##### TRANSFORMER ARCHITECTURE #####

class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(1)
            wei = wei.masked_fill(mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        out = torch.cat([h(x, key_padding_mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x_ln1 = self.ln1(x)
        if key_padding_mask is not None:
            x_ln1 = x_ln1 * key_padding_mask.unsqueeze(-1)
        x = x + self.sa(x_ln1, key_padding_mask)
        x_ln2 = self.ln2(x)
        if key_padding_mask is not None:
            x_ln2 = x_ln2 * key_padding_mask.unsqueeze(-1)
        x = x + self.ffwd(x_ln2)
        if key_padding_mask is not None:
            x = x * key_padding_mask.unsqueeze(-1)
        return x

class GPTmodel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        block_size: int,
        dropout: float,
        pad_token: int
    ):
        super().__init__()
        self.pad_token = pad_token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx=pad_token)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_ids = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos_ids)
        x = tok_emb + pos_emb.unsqueeze(0)
        if key_padding_mask is not None:
            x = x * key_padding_mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        logits[..., self.pad_token] = -1e9
        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            loss = F.cross_entropy(logits_flat, targets.view(-1), ignore_index=self.pad_token)
        return logits, loss
    
    def generate(self, idx, max_new_tokens, block_size) :
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if _ == max_new_tokens - 1:
              # if we are at the last step, return the sequence
              return idx
            if idx_next == 0:
              return idx
    

    def generate_prob(self, idx, max_new_tokens, block_size, decode_fn, fasta_path="output.fasta"):
        # helper to save sequences + log-probs to FASTA and return
        def _save_and_return(sequences, all_log_probs):
            seqs = sequences.cpu().tolist()
            with open(fasta_path, "a") as f:
                for i, (seq, logs) in enumerate(zip(seqs, all_log_probs)):
                    seq_str     = decode_fn(seq).strip()
                    total_logp  = sum(logs)
                    mean_logp   = total_logp / len(logs) if logs else 0.0
                    f.write(f">sequence_{i} log_prob={total_logp:.4f} mean_log_prob={mean_logp:.4f}\n")
                    f.write(f"{seq_str}\n")
            return sequences

        # initialize per-batch log-prob storage
        B = idx.size(0)
        all_log_probs = [[] for _ in range(B)]

        for step in range(max_new_tokens):
        # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # record log-probs
            p_next       = probs.gather(1, idx_next).squeeze(1)
            log_p_next   = torch.log(p_next)
            for i, lp in enumerate(log_p_next.tolist()):
                all_log_probs[i].append(lp)

            # append sampled index to the running sequence

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            # if last step OR we've sampled an EOS (0) anywhere, save+return
            if step == max_new_tokens - 1 or idx_next.eq(0).any():
                return _save_and_return(idx, all_log_probs)

    # (should never reach here, but for completeness)
        return _save_and_return(idx, all_log_probs)


def set_dropout(model, new_p):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = new_p


########### LOSS FUNCTION ###########

def reint_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    Rb: torch.Tensor,
    pad_token: int
) -> torch.Tensor:
    B, T, V = logits.size()
    loss_flat = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1),
        ignore_index=pad_token,
        reduction='none'
    )
    loss = loss_flat.view(B, T) * Rb.unsqueeze(-1)
    mask = targets != pad_token
    return loss[mask].mean()



def reint_ppo_loss(logits,
                   legacy_logits,
                   targets,
                   Rb,
                   batch_size,
                   pad_token,
                   clip_eps: float = 0.2):
    

    # logits:          [B * T, C] current policy logits
    # legacy_logits:   [B * T, C] old (frozen) policy logits
    # targets:         [B, T] token IDs, with pad_token where padding
    # Rb:              [B] tensor of +1 (good) / -1 (bad)
    # pad_token:       int, the ID to ignore
    # clip_eps:        PPO clipping ε (e.g. 0.2)

    # Returns:
    #   scalar PPO loss (to .backward())

    logits = logits.view(batch_size, -1, logits.shape[-1])  # (B, T, C)
    legacy_logits = legacy_logits.view(batch_size, -1, legacy_logits.shape[-1])

    logp_new = F.log_softmax(logits, dim=-1)         # (B, T, C)
    logp_old = F.log_softmax(legacy_logits, dim=-1)  # (B, T, C)

    ratiop = torch.exp(logp_new - logp_old)  # (B, T, C)

    ratiop = ratiop.gather(dim=-1, index=targets.unsqueeze(-1))  # (B, T, 1)  ratio at the target token

    unclipped = ratiop.view(batch_size, -1) * Rb.unsqueeze(-1)  # (B, T)  ratio * reward
    clipped = torch.clamp(
        ratiop.view(batch_size, -1), 1 - clip_eps, 1 + clip_eps
    ) * Rb.unsqueeze(-1)  # (B, T)  clipped ratio * reward

    loss = -torch.min(unclipped, clipped)  # (B, T)  min of unclipped and clipped
    PPO_loss = loss.view(-1)[targets.view(-1) != pad_token].mean()

    return PPO_loss  # scalar loss to .backward()



def dkl_between_logits(logits, legacy_logits):
    """
    Compute average KL divergence D_KL(P || Q) over (Batch, Tokens),
    where P = softmax(logits), Q = softmax(legacy_logits).
    
    Args:
        logits: Tensor of shape (B, T, C)
        legacy_logits: Tensor of shape (B, T, C)

    Returns:
        Scalar tensor: mean KL divergence over batch and time.
    """
    # Ensure float type
    logits = logits.float()
    legacy_logits = legacy_logits.float()

    # Compute log-probs
    log_p = F.log_softmax(logits, dim=-1)         # log P
    log_q = F.log_softmax(legacy_logits, dim=-1)  # log Q

    # Get probs from log_p
    p = log_p.exp()  # P

    # Compute D_KL(P || Q) per token: sum_c P * (logP - logQ)
    dkl = (p * (log_p - log_q)).sum(dim=-1)  # shape: (B, T)

    # Average over batch and time
    return dkl.mean()




########### LOAD DATASETS & BATCH & EVALUATE ############

def load_dataset(
    seq_path: str,
    label_path: str,
    block_size: int,
    train_size: int,
    pad_token: int,
    encode_fn: Callable[[str], List[int]],
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Loads sequences and labels, encodes/pads to block_size, shuffles,
    and splits into train/validation with separate good/bad for val.
    Returns dict with keys:
      'train', 'train_good', 'train_bad', 'val', 'val_good', 'val_bad'
    Each value is a dict with 'data' and 'labels'.
    """
    # Read sequences
    with open(seq_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    seqs = [lines[i].strip() for i in range(1, len(lines), 2) if lines[i].strip()]
    # Read labels
    if label_path == ".":
        labels = [1] * len(seqs)
    else:
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [int(x) for x in f.read().splitlines()]
        assert len(seqs) == len(labels), "Sequences and labels must match"

    # Encode & pad helper
    def pad_encode(seq: str) -> torch.Tensor:
        txt = f"\n{seq}\n"
        ids = encode_fn(txt)
        if len(ids) < block_size:
            ids += [pad_token] * (block_size - len(ids))
        else:
            ids = ids[:block_size]
        return torch.tensor(ids, dtype=torch.long)

    data = [pad_encode(s) for s in seqs]

    # Split train/val
    val_data = data[train_size:]
    val_labels = labels[train_size:]
    train_data = data[:train_size]
    train_labels = labels[:train_size]

    num_train = len(train_data)
    num_val = len(val_data)
    pos_train = sum(1 for l in train_labels if l == 1)
    neg_train = num_train - pos_train
    pos_val = sum(1 for l in val_labels if l == 1)
    neg_val = num_val - pos_val
    print(f"Train sequences: {num_train}, Validation sequences: {num_val}")
    print(f"Train positives: {pos_train} ({100*pos_train/num_train:.2f}%), "
          f"negatives: {neg_train} ({100*neg_train/num_train:.2f}%)")
    print(f"Val positives: {pos_val} ({100*pos_val/num_val:.2f}%), "
          f"negatives: {neg_val} ({100*neg_val/num_val:.2f}%)")

    # Helper to split good/bad
    def split_by_label(data_list, label_list):
        good = [(d, l) for d, l in zip(data_list, label_list) if l == 1]
        bad  = [(d, l) for d, l in zip(data_list, label_list) if l != 1]
        return {
            'good': {'data': [d for d,_ in good], 'labels': [l for _,l in good]},
            'bad':  {'data': [d for d,_ in bad],  'labels': [l for _,l in bad]}
        }

    train_split = {'data': train_data, 'labels': train_labels}
    val_split   = {'data': val_data,   'labels': val_labels}
    train_gb = split_by_label(train_data, train_labels)
    val_gb   = split_by_label(val_data,   val_labels)

    return {
        'train':       train_split,
        'train_good':  train_gb['good'],
        'train_bad':   train_gb['bad'],
        'val':         val_split,
        'val_good':    val_gb['good'],
        'val_bad':     val_gb['bad'],
    }


def get_batch(
    split: str,
    datasets: Dict[str, Dict[str, List[torch.Tensor]]],
    batch_size: int,
    block_size: int,
    pad_token: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds a batch for the given split key.
    Returns x, y, pad_mask, R (labels, negatives downweighted).
    """
    data_list = datasets[split]['data']
    labels = datasets[split]['labels']

    idxs = torch.randint(0, len(data_list), (batch_size,)).tolist()
    starts = [
        torch.randint(0, max(1, data_list[i].size(0) - block_size - 1), ()).item()
        for i in idxs
    ]
    x = torch.stack([data_list[i][s : s + block_size] for i,s in zip(idxs, starts)], dim=0)
    y = torch.stack([data_list[i][s + 1 : s + block_size + 1] for i,s in zip(idxs, starts)], dim=0)
    
    pad_col = torch.full((y.size(0), 1), pad_token, dtype=y.dtype, device=y.device)
    y = torch.cat([y, pad_col], dim=1)

    pad_mask = (x != pad_token)
    R = torch.tensor([labels[i] for i in idxs], dtype=torch.float)

    return x.to(device), y.to(device), pad_mask.to(device), R.to(device)

def estimate_evaluation_losses(
    model,
    datasets: Dict[str, Dict[str, List[torch.Tensor]]],
    eval_iters: int,
    batch_size: int,
    block_size: int,
    pad_token: int,
    device: torch.device,
    mode = "legacy"
) -> Dict[str, float]:
    """
    Computes average probabilities (losses with R=1) for:
      - 'train_good'
      - 'train_bad'
      - 'val_good'
      - 'val_bad'
    """
    if mode == "legacy":
         keys = ['train_good', 'val_good']
    else:
         keys = ['train_good', 'train_bad', 'val_good', 'val_bad']

    model.eval()
    sums = {k: 0.0 for k in keys}
    for _ in range(eval_iters):
        for key in keys:
            x, y, mask, _ = get_batch(key, datasets, batch_size, block_size, pad_token, device)
            logits, loss = model(x, mask)
            # override R to all ones for equal weighting
            R1 = torch.ones(x.size(0), device=device)
            sums[key] += reint_loss(logits, y, R1, pad_token).item()
    model.train()
    return {k: sums[k] / eval_iters for k in keys}

