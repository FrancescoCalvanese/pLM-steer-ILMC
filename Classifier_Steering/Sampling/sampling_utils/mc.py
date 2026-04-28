import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext

__all__ = [
    "generate_block",
    "continue_generation",
    "resample_from_last_k",
    "metropolis_acceptance",
    "score_sequences",
    "print_fasta",
    "load_fasta"
]

def generate_block(model, beta_sampling, N, num_tokens, max_context, device, bos_eos_token, pad_token):
    """
    Generate first num_tokens tokens for N sequences in a highly optimized manner.
    
    Args:
        model: The language model
        beta_sampling: Temperature parameter for sampling (scales logits)
        N: Number of sequences to generate
        num_tokens: Number of tokens to generate per sequence
        max_context: Maximum context window size for the model
        device: Device to run on (should be 'cuda' for speed)
        bos_eos_token: Beginning/End of sequence token ID (same index)
        pad_token: Padding token ID
    
    Returns:
        sequences: (N, num_tokens+1) tensor containing [BOS, token_1, ..., token_{num_tokens}]
        per_token_legacy_log_probs: (N, num_tokens) tensor of per token log probabilities (beta=1.0)
        per_token_sampling_log_probs: (N, num_tokens) tensor of per token log probabilities (beta=beta_sampling)
    """
    model.eval()
    
    # Initialize sequences with BOS token: (N, 1)
    sequences = torch.full((N, 1), bos_eos_token, dtype=torch.long, device=device)
    
    # Pre-allocate output tensors for speed
    all_tokens = torch.zeros((N, num_tokens), dtype=torch.long, device=device)
    per_token_legacy_log_probs = torch.zeros((N, num_tokens), dtype=torch.float32, device=device)
    per_token_sampling_log_probs = torch.zeros((N, num_tokens), dtype=torch.float32, device=device)
    
    # Track which sequences are finished (hit EOS token)
    finished = torch.zeros(N, dtype=torch.bool, device=device)
    
    with torch.inference_mode():
        for step in range(num_tokens):
            # Crop to last max_context tokens (efficient slicing)
            start_idx = max(0, sequences.size(1) - max_context)
            idx_cond = sequences[:, start_idx:]
            
            # Forward pass: fully batched across all N sequences
            # Assumes model returns a dict with 'logits'
            logits = model(idx_cond)['logits']
            logits = logits[:, -1, :]  # (N, vocab_size)
            
            # Compute probabilities
            probs_sampling = F.softmax(beta_sampling * logits, dim=-1)  # (N, vocab_size)
            probs_legacy = F.softmax(logits, dim=-1)  # (N, vocab_size)

            # Sample next tokens for all sequences at once
            next_tokens = torch.multinomial(probs_sampling, num_samples=1).squeeze(1)  # (N,)

            # Force PAD token for sequences that are already finished
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token), next_tokens)
            
            # Get log probabilities of sampled tokens
            legacy_log_probs = torch.log(probs_legacy.gather(1, next_tokens.unsqueeze(1)).squeeze(1) + 1e-12)  # (N,)
            sampling_log_probs = torch.log(probs_sampling.gather(1, next_tokens.unsqueeze(1)).squeeze(1) + 1e-12)  # (N,)
            
            # For finished sequences, set log prob to zero
            legacy_log_probs = torch.where(finished, torch.zeros_like(legacy_log_probs), legacy_log_probs)          
            sampling_log_probs = torch.where(finished, torch.zeros_like(sampling_log_probs), sampling_log_probs)
            
            # Store tokens and per token log probs
            all_tokens[:, step] = next_tokens
            per_token_legacy_log_probs[:, step] = legacy_log_probs
            per_token_sampling_log_probs[:, step] = sampling_log_probs

            # Append to sequences
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            
            # Update finished flags (check if we just generated EOS)
            newly_finished = (next_tokens == bos_eos_token) & ~finished
            finished = finished | newly_finished
            
            # Early stopping if all sequences finished
            if finished.all():
                # Fill remaining positions with PAD
                if step < num_tokens - 1:
                    all_tokens[:, step+1:] = pad_token
                    # Per token log probs remain zero for padded positions
                break
    
    return sequences, per_token_legacy_log_probs, per_token_sampling_log_probs


def continue_generation(sequences, per_token_legacy_log_probs, per_token_sampling_log_probs, 
                        model, beta_sampling, num_new_tokens, max_context, device, bos_eos_token, pad_token):
    """
    Continue generation from existing sequences, adding num_new_tokens more tokens.
    """
    model.eval()
    
    N = sequences.size(0)
    # L = sequences.size(1) (Unused variable removed)
    
    # Move tensors to the correct device if needed
    sequences = sequences.to(device)
    per_token_legacy_log_probs = per_token_legacy_log_probs.to(device)
    per_token_sampling_log_probs = per_token_sampling_log_probs.to(device)
    
    # Pre-allocate space for new tokens and per token log probs
    new_tokens_tensor = torch.zeros((N, num_new_tokens), dtype=torch.long, device=device)
    new_per_token_legacy_log_probs = torch.zeros((N, num_new_tokens), dtype=torch.float32, device=device)
    new_per_token_sampling_log_probs = torch.zeros((N, num_new_tokens), dtype=torch.float32, device=device)
    
    # Check which sequences are already finished (contain EOS after BOS)
    # Look for EOS in positions 1 onwards (skip BOS at position 0)
    finished = torch.zeros(N, dtype=torch.bool, device=device)
    for i in range(N):
        if (sequences[i, 1:] == bos_eos_token).any():
            finished[i] = True
    
    with torch.inference_mode():
        for step in range(num_new_tokens):
            # Crop to last max_context tokens (efficient slicing)
            start_idx = max(0, sequences.size(1) - max_context)
            idx_cond = sequences[:, start_idx:]
            
            # Forward pass: fully batched across all N sequences
            logits = model(idx_cond)['logits']
            logits = logits[:, -1, :]  # (N, vocab_size)
            
            # Compute probabilities with different betas
            probs_sampling = F.softmax(beta_sampling * logits, dim=-1)  # (N, vocab_size)
            probs_legacy = F.softmax(logits, dim=-1)  # (N, vocab_size)
            
            # Sample next tokens for all sequences at once (using sampling distribution)
            next_tokens = torch.multinomial(probs_sampling, num_samples=1).squeeze(1)  # (N,)
            
            # Force PAD token for sequences that are already finished
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token), next_tokens)
            
            # Get log probabilities of sampled tokens from both distributions
            legacy_log_probs = torch.log(probs_legacy.gather(1, next_tokens.unsqueeze(1)).squeeze(1) + 1e-12)  # (N,)
            sampling_log_probs = torch.log(probs_sampling.gather(1, next_tokens.unsqueeze(1)).squeeze(1) + 1e-12)  # (N,)
            
            # For finished sequences, set log prob to zero
            legacy_log_probs = torch.where(finished, torch.zeros_like(legacy_log_probs), legacy_log_probs)
            sampling_log_probs = torch.where(finished, torch.zeros_like(sampling_log_probs), sampling_log_probs)
            
            # Store tokens and per token log probs
            new_tokens_tensor[:, step] = next_tokens
            new_per_token_legacy_log_probs[:, step] = legacy_log_probs
            new_per_token_sampling_log_probs[:, step] = sampling_log_probs
            
            # Append to sequences
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            
            # Update finished flags (check if we just generated EOS)
            newly_finished = (next_tokens == bos_eos_token) & ~finished
            finished = finished | newly_finished
            
            # Early stopping if all sequences finished
            if finished.all():
                # Fill remaining positions with PAD
                if step < num_new_tokens - 1:
                    new_tokens_tensor[:, step+1:] = pad_token
                    # Per token log probs remain zero for padded positions
                break
    
    # Concatenate the new per token log probs to the original ones
    per_token_legacy_log_probs = torch.cat([per_token_legacy_log_probs, new_per_token_legacy_log_probs], dim=1)
    per_token_sampling_log_probs = torch.cat([per_token_sampling_log_probs, new_per_token_sampling_log_probs], dim=1)
    
    return sequences, per_token_legacy_log_probs, per_token_sampling_log_probs




def resample_from_last_k(sequences, per_token_legacy_log_probs, per_token_sampling_log_probs, 
                         model, beta_sampling, max_context, device, bos_eos_token, pad_token, k):
    """
    Resample the last m tokens for all sequences, where m is chosen randomly 
    between 1 and k (shared across the batch).
    """
    model.eval()
    
    N = sequences.size(0)
    L = sequences.size(1)
    
    # Move tensors to device
    sequences = sequences.to(device)
    per_token_legacy_log_probs = per_token_legacy_log_probs.to(device)
    per_token_sampling_log_probs = per_token_sampling_log_probs.to(device)
    
    # Clone to create new containers
    new_sequences = sequences.clone()
    new_per_token_legacy_log_probs = per_token_legacy_log_probs.clone()
    new_per_token_sampling_log_probs = per_token_sampling_log_probs.clone()
    
    # Calculate actual lengths (ignoring padding)
    non_pad_mask = (sequences != pad_token)
    actual_lengths = non_pad_mask.sum(dim=1)  # (N,)
    
    # --- CHANGED SECTION ---
    # Pick a single random 'depth' m to resample, shared by all chains.
    # m is between 1 (resample only the very last token) and k (resample last k tokens).
    m = torch.randint(1, k + 1, (1,), device=device).item()
    
    # Determine start position for each sequence based on this shared m
    # Start = Length - m, clamped to min=1 (never resample BOS at index 0)
    start_positions = torch.clamp(actual_lengths - m, min=1)  # (N,)
    # ---------------------
    
    # Loop from the earliest required position to the end
    min_loop_start = start_positions.min().item()
    
    # Track which sequences have finished generation (hit EOS) during resampling
    finished = torch.zeros(N, dtype=torch.bool, device=device)
    
    with torch.inference_mode():
        for pos in range(min_loop_start, L):
            # Mask: Which sequences are currently in their "resampling window"?  
            active_mask = (torch.tensor(pos, device=device) >= start_positions)
            
            # If no sequences are active yet, continue
            if not active_mask.any():
                continue

            # Get context (use new_sequences to allow autoregressive generation)
            start_idx = max(0, pos - max_context)
            idx_cond = new_sequences[:, start_idx:pos]
            
            # Forward pass
            logits = model(idx_cond)['logits']
            logits = logits[:, -1, :]
            
            # Compute probabilities
            probs_sampling = F.softmax(beta_sampling * logits, dim=-1)
            probs_legacy = F.softmax(logits, dim=-1)
            
            # Sample candidates
            candidate_tokens = torch.multinomial(probs_sampling, num_samples=1).squeeze(1)
            
            # Determine the actual token to keep at this position
            original_tokens = new_sequences[:, pos]
            
            # Apply 'finished' padding logic only to active sequences
            next_tokens = torch.where(finished, torch.full_like(candidate_tokens, pad_token), candidate_tokens)
            
            # Finalize tokens: if not active, revert to original
            final_tokens = torch.where(active_mask, next_tokens, original_tokens)
            new_sequences[:, pos] = final_tokens

            # Calculate log probs for the tokens at this position
            curr_legacy_log = torch.log(probs_legacy.gather(1, final_tokens.unsqueeze(1)).squeeze(1) + 1e-12)
            curr_sampling_log = torch.log(probs_sampling.gather(1, final_tokens.unsqueeze(1)).squeeze(1) + 1e-12)
            
            # Zero out log probs for finished sequences
            curr_legacy_log = torch.where(finished, torch.zeros_like(curr_legacy_log), curr_legacy_log)
            curr_sampling_log = torch.where(finished, torch.zeros_like(curr_sampling_log), curr_sampling_log)

            # Update per token log probs only where active
            new_per_token_legacy_log_probs[:, pos-1] = torch.where(
                active_mask, 
                curr_legacy_log, 
                new_per_token_legacy_log_probs[:, pos-1]
            )
            new_per_token_sampling_log_probs[:, pos-1] = torch.where(
                active_mask, 
                curr_sampling_log, 
                new_per_token_sampling_log_probs[:, pos-1]
            )

            # Update finished status
            newly_finished = (final_tokens == bos_eos_token) & active_mask & ~finished
            finished = finished | newly_finished
            
            # Early stopping if all active sequences are finished
            if active_mask.all() and finished.all():
                if pos < L - 1:
                    new_sequences[:, pos+1:] = pad_token
                    # Per token log probs remain zero for padded positions
                    new_per_token_legacy_log_probs[:, pos:] = 0.0
                    new_per_token_sampling_log_probs[:, pos:] = 0.0
                break

    return new_sequences, new_per_token_legacy_log_probs, new_per_token_sampling_log_probs


def metropolis_acceptance(energy_0, energy_1, device):
    """
    Perform Metropolis-Hastings acceptance test between two sets of sequences based on energies.
    
    For symmetric proposals (where q(x'|x) = q(x|x')), the acceptance probability simplifies to:
    alpha = min(1, exp(-(energy_1 - energy_0)))
    
    Args:
        energy_0: (N,) tensor of energies for current states
        energy_1: (N,) tensor of energies for proposed states
        device: Device to run on
    
    Returns:
        accept_mask: (N,) boolean tensor, True where proposal is accepted, False otherwise
    """
    N = energy_0.size(0)

    # Move everything to device
    energy_0 = energy_0.to(device)
    energy_1 = energy_1.to(device)

    # Compute log acceptance ratio: log(alpha) = -(energy_1 - energy_0)
    log_acceptance_ratio = -(energy_1 - energy_0)
    
    # Compute acceptance probability: alpha = min(1, exp(log_acceptance_ratio))
    log_acceptance_prob = torch.clamp(log_acceptance_ratio, max=0.0)
    acceptance_prob = torch.exp(log_acceptance_prob)  # (N,)
    
    # Draw random uniform samples
    uniform_samples = torch.rand(N, device=device)  # (N,)
    
    # Accept if uniform sample < acceptance probability
    accept_mask = uniform_samples < acceptance_prob  # (N,) boolean tensor
    
    return accept_mask

@torch.inference_mode()
def score_sequences(sequences, model, pad_token=None):
    """
    Score a list of tokenized sequences using the provided autoregressive model,
    returning the negative log-likelihood (negative log probability) PER SEQUENCE.  
    
    Args:
        sequences: (Batch_Size, Seq_Len) tensor of token IDs
        model: The autoregressive language model
        pad_token: Optional padding token ID to exclude from loss calculation
    
    Returns:
        torch.Tensor: A tensor of shape [Batch_Size] containing the negative log probability 
                      (total loss) for each sequence.  
    """
    model.eval()
    
    # Assuming ctx is defined globally as in your original snippet
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with ctx:
            with sdpa_kernel(backends=[SDPBackend. FLASH_ATTENTION]):
                # 1. Forward pass: Get logits only.   
                # Input is sequences[:, :-1] to predict sequences[:, 1:]
                outputs = model(sequences[:, :-1])
    except (ImportError, NameError):
        # If sdpa_kernel is not available or ctx is not defined, run without it
        outputs = model(sequences[:, :-1])
            
    # 2. Extract logits.  Shape: [Batch_Size, Seq_Len, Vocab_Size]
    logits = outputs["logits"]
    
    # 3. Define targets (shifted by 1).  Shape: [Batch_Size, Seq_Len]
    targets = sequences[:, 1:]. contiguous()

    # 4. Compute Cross Entropy manually with reduction='none'
    # PyTorch CrossEntropy expects logits as [Batch, Vocab, Seq_Len], 
    # so we transpose dimensions 1 and 2.
    loss_per_token = F.cross_entropy(
        logits.transpose(1, 2), 
        targets, 
        reduction='none',
        ignore_index=pad_token if pad_token is not None else -100
    ) 
    # loss_per_token shape is now [Batch_Size, Seq_Len]

    # 5. Sum the loss over the time dimension (dim=1) to get total negative log prob per sequence
    # This gives us -log P(sequence) for each sequence
    log_prob_per_sequence = -loss_per_token.sum(dim=1)
    
    return log_prob_per_sequence.to(torch.float32)


def print_fasta(sequences, pad_token, decode):
    """
    Print sequences along with their distances.
    
    Args:
        sequences: (N, L) tensor of sequences
        distance_list: (N,) tensor of distances
        pad_token: Padding token ID
        decode: Function to decode token IDs to strings
    """
    for i in range(sequences.size(0)):
        seq = sequences[i]
        # Remove padding tokens for printing
        seq_nopad = seq[seq != pad_token]
        decoded_seq = decode(seq_nopad.tolist())
        print(f">Sequence_{i}: {decoded_seq[0:-1]}")


def load_fasta(file_path, encode, bos_eos_token, pad_token, device, max_length=None):
    """
    Loads sequences, adds BOS/EOS tokens, pads them, and returns a tensor.
    
    Structure: [BOS] + [Sequence] + [EOS] + [Padding...]

    Args:
        file_path (str): Path to the .fasta file.
        encode (callable): Function that takes a string and returns a list of token IDs.
        bos_eos_token (int): Token ID used for both beginning and end of sequence.
        pad_token (int): Token ID used for padding.
        device (torch.device): Device to load the tensor onto.
        max_length (int, optional): Force final tensor width. 
                                    If None, fits to the longest sequence + 2.

    Returns:
        torch.Tensor: A tensor of shape (Num_Sequences, Max_Length).
    """
    sequences = []
    current_seq = []
    
    # 1. Read and Parse FASTA
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        
        if current_seq:
            sequences.append("".join(current_seq))

    if not sequences:
        print(f"Warning: No sequences found in {file_path}")
        return torch.empty(0, device=device)

    # 2. Encode and Add BOS/EOS
    processed_seqs = []
    for seq in sequences:
        # Encode string to integers
        encoded_data = encode(seq)
        
        # Ensure list format
        if isinstance(encoded_data, torch.Tensor):
            encoded_data = encoded_data.tolist()
            
        # Add BOS and EOS
        full_seq = [bos_eos_token] + encoded_data + [bos_eos_token]
        processed_seqs.append(full_seq)

    # 3. Determine Max Length
    if max_length is None:
        max_length = max(len(s) for s in processed_seqs)

    # 4. Pad, Truncate, and Stack
    tensor_list = []
    for seq in processed_seqs:
        # Truncate if longer than max_length
        if len(seq) > max_length:
            # We truncate, but usually try to keep the EOS if possible.
            # Here is simple right-truncation:
            seq = seq[:max_length] 
        
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        
        # Pad if shorter than max_length
        if len(seq) < max_length:
            padding = torch.full((max_length - len(seq),), pad_token, dtype=torch.long)
            seq_tensor = torch.cat([seq_tensor, padding])
            
        tensor_list.append(seq_tensor)

    return torch.stack(tensor_list).to(device)
