import torch
import torch.nn.functional as F

from plm_steer.sampling import BaseSampler, MCMCState
from plm_steer.model.transformer import GPTTransformer


class GPTSampler(BaseSampler):

    def __init__(
        self,
        model: GPTTransformer,
        beta_sampling: float,
        device: str,
        eos_token_id: int,
        pad_token_id: int,
    ):
        super().__init__(model, beta_sampling, device)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    @torch.inference_mode()
    def _generation_step(
        self,
        sequences: torch.LongTensor,
        finished_mask: torch.BoolTensor | None = None,
        active_mask: torch.BoolTensor | None = None,
        original_tokens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor]:

        if finished_mask is None:
            finished_mask = torch.zeros(sequences.size(0), dtype=torch.bool, device=self.device)

        logits = self.model(sequences)["logits"]
        logits = logits[:, -1, :]

        # Compute probabilities
        probs_sampling = F.softmax(self.beta_sampling * logits, dim=-1)
        probs_legacy = F.softmax(logits, dim=-1)

        next_tokens = torch.multinomial(probs_sampling, num_samples=1).squeeze(1)
        next_tokens = torch.where(finished_mask, self.pad_token_id, next_tokens)

        if original_tokens is not None:
            if active_mask is None:
                raise ValueError("active_mask must be provided when original_tokens is used.")
            # apply resampling logic
            next_tokens = torch.where(active_mask, next_tokens, original_tokens)

        # Get log probabilities of sampled tokens
        legacy_log_probs = torch.log(
            probs_legacy.gather(1, next_tokens.unsqueeze(1)).squeeze(1) + 1e-12
        )
        sampling_log_probs = torch.log(
            probs_sampling.gather(1, next_tokens.unsqueeze(1)).squeeze(1) + 1e-12
        )

        # For finished sequences, set log prob to zero
        legacy_log_probs = torch.where(finished_mask, 0, legacy_log_probs)
        sampling_log_probs = torch.where(finished_mask, 0, sampling_log_probs)

        return next_tokens, legacy_log_probs, sampling_log_probs

    def generate_block(self, input_state: torch.Tensor | MCMCState, block_size: int) -> MCMCState:
        input_state = input_state.to(self.device)
        is_first_block = isinstance(input_state, torch.Tensor)
        sequences = input_state if is_first_block else input_state.sequences
        batch_size = sequences.size(0)

        # Initialize containers
        block_tokens = torch.full(
            (batch_size, block_size), self.pad_token_id, dtype=torch.long, device=self.device
        )
        legacy_log_probs = torch.zeros((batch_size, block_size), device=self.device)
        sampling_log_probs = torch.zeros((batch_size, block_size), device=self.device)

        if is_first_block:
            finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        else:
            finished_mask = (sequences[:, 1:] == self.eos_token_id).any(dim=1)

        for step in range(block_size):
            next_token, next_legacy_log_probs, next_sampling_log_probs = self._generation_step(
                sequences, finished_mask
            )

            block_tokens[:, step] = next_token
            legacy_log_probs[:, step] = next_legacy_log_probs
            sampling_log_probs[:, step] = next_sampling_log_probs

            # Update finished flags (check if we just generated EOS)
            newly_finished = (next_token == self.eos_token_id) & ~finished_mask
            finished_mask = finished_mask | newly_finished
            
            if finished_mask.all():
                break
            
        sequences = torch.cat([sequences, block_tokens], dim=1)
        if not is_first_block:
            legacy_log_probs = torch.cat([input_state.legacy_log_probs, legacy_log_probs], dim=1)
            sampling_log_probs = torch.cat(
                [input_state.sampling_log_probs, sampling_log_probs], dim=1
            )

        return MCMCState(sequences, legacy_log_probs, sampling_log_probs)

    def resample(self, state: MCMCState, window_size: int) -> MCMCState:

        state = state.to(self.device)
        batch_size, seq_len = state.sequences.size()
        original_prompt_len = seq_len - state.sampling_log_probs.size(1)

        # Clone to create new containers
        new_sequences = state.sequences.clone()
        new_per_token_legacy_log_probs = state.legacy_log_probs.clone()
        new_per_token_sampling_log_probs = state.sampling_log_probs.clone()

        # Calculate actual lengths (ignoring padding)
        actual_lengths = state.sequences.ne(self.pad_token_id).sum(dim=1)
        # Pick a single random 'depth' m to resample, shared by all chains.
        # m is between 1 (resample only the very last token) and k (resample last k tokens).
        m = torch.randint(1, window_size + 1, (1,), device=self.device).item()

        # Determine start position for each sequence based on this shared m
        start_positions = torch.clamp(actual_lengths - m, min=original_prompt_len)
        min_loop_start = start_positions.min().item()

        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # finished_mask = (new_sequences[:, 1:] == self.eos_token_id).any(dim=1)

        for pos in range(min_loop_start, seq_len):
            active_mask = (pos >= start_positions)  # & (~finished_mask)

            if not active_mask.any():
                continue  # No sequences to resample at this position

            next_tokens, next_legacy_log_probs, next_sampling_log_probs = self._generation_step(
                new_sequences[:, :pos],
                finished_mask=finished_mask,
                active_mask=active_mask,
                original_tokens=state.sequences[:, pos],
            )

            # resampling logic already applied in _generation_step
            new_sequences[:, pos] = next_tokens

            # Update per token log probs only where active
            log_prob_idx = pos - original_prompt_len
            new_per_token_legacy_log_probs[:, log_prob_idx] = torch.where(
                active_mask, next_legacy_log_probs, new_per_token_legacy_log_probs[:, log_prob_idx]
            )
            new_per_token_sampling_log_probs[:, log_prob_idx] = torch.where(
                active_mask,
                next_sampling_log_probs,
                new_per_token_sampling_log_probs[:, log_prob_idx],
            )

            # Update finished flags (check if we just generated EOS)
            newly_finished = (next_tokens == self.eos_token_id) & ~finished_mask
            finished_mask = finished_mask | newly_finished
            
            if finished_mask.all() and active_mask.all():  # to check
                if pos < seq_len - 1:
                    new_sequences[:, pos + 1 :] = self.pad_token_id
                    new_per_token_legacy_log_probs[:, log_prob_idx + 1 :] = 0
                    new_per_token_sampling_log_probs[:, log_prob_idx + 1 :] = 0
                break

        return MCMCState(
            new_sequences, new_per_token_legacy_log_probs, new_per_token_sampling_log_probs
        )
