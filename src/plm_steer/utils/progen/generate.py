"""Generation utils for ProGen3"""

import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.generation import (
    GenerateDecoderOnlyOutput,
    LogitsProcessor,
    LogitsProcessorList,
    GenerationConfig,
)

try:
    from progen3.modeling import ProGen3ForCausalLM
    from progen3.batch_preparer import ProGen3BatchPreparer
    from progen3.tools.utils import get_progen3_model
except ImportError:
    raise ImportError("Progen3 package not found. Please install it first.")

from plm_steer.sampling import BaseSampler, MCMCState

logging.getLogger("transformers").setLevel(logging.ERROR)


def load_progen3_model(model_name: str, use_fsdp: bool = False) -> ProGen3ForCausalLM:
    """
    Load ProGen3 model on correct available device from model name or local path.
    """
    model = get_progen3_model(model_name, use_fsdp)
    return model.eval()


class LengthRangeLogitsProcessor(LogitsProcessor):
    """Enforce min and max total length (prompt + new during generation."""

    def __init__(self, min_length: int, max_length: int, cterm_token_id: int, nterm_token_id: int, pad_token_id: int):

        self.min_length = min_length
        self.max_length = max_length
        self.cterm_token_id = cterm_token_id
        self.nterm_token_id = nterm_token_id
        self.pad_token_id = pad_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        cur_len = input_ids.size(1)  # total length including prompt
        if cur_len < self.min_length:
            # Suppress CTERM token  (maybe also NTERM token should be suppressed?) only if there is
            # at least one token, except CTERM token, with prob > -inf
            scores[:, self.pad_token_id] = -float("inf")
            scores[:, self.cterm_token_id] = -float("inf")
            if cur_len > 2:  # allow NTERM only at beginning
                scores[:, self.nterm_token_id] = -float("inf")
            inf_mask = torch.isinf(scores).all(dim=-1)
            if inf_mask.any():
                print("Warning: All logits are -inf for some sequences during length enforcement.")
                scores[inf_mask, self.cterm_token_id] = 0.0  # allow CTERM in this case
        if cur_len >= self.max_length:
            # Suppress all tokens except CTERM (To check that CTermLogitsProcessor acts later)
            scores.fill_(-float("inf"))
            scores[:, self.cterm_token_id] = 0.0
        return scores


class CTermLogitsProcessor(LogitsProcessor):
    """Enforce C-term - EOS - PAD sequence during generation."""

    def __init__(
        self,
        mask_finished: torch.BoolTensor,
        pad_token_id: int,
        eos_token_id: int,
        cterm_token_id: int,
    ):
        self.mask_finished = mask_finished
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.cterm_token_id = cterm_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Check the entire history for the sequence, not just the last token
        # This prevents "leaks" if the processor is called during a re-start or resampling
        has_eos = (input_ids == self.eos_token_id).any(dim=1)
        has_cterm = (input_ids == self.cterm_token_id).any(dim=1)

        # 1. Update the finished mask
        self.mask_finished |= has_eos

        # 2. Identify sequences that JUST generated a C-term and need an EOS
        # (They have a '2' but don't have an 'eos' yet)
        needs_eos_transition = has_cterm & ~has_eos

        # 3. Apply constraints
        # Priority 1: If finished (already has EOS), force PAD
        scores[~self.mask_finished, self.pad_token_id] = -float("inf")
        inf_mask = torch.isinf(scores).all(dim=-1)
        if inf_mask.any():
            # print("Warning: All logits are -inf for some sequences during length enforcement.")
            scores[
                inf_mask & (~self.mask_finished) & (~needs_eos_transition), self.cterm_token_id
            ] = 0.0  # allow CTERM in this case
        if self.mask_finished.any():
            scores[self.mask_finished, :] = -float("inf")
            scores[self.mask_finished, self.pad_token_id] = 0.0

        # Priority 2: If just hit C-term, force EOS next
        if needs_eos_transition.any():
            scores[needs_eos_transition, :] = -float("inf")
            scores[needs_eos_transition, self.eos_token_id] = 0.0

        return scores


class ProGen3Sampler(BaseSampler):
    """MCMC Sampler wrapper for ProGen3 model."""

    def __init__(
        self,
        model: ProGen3ForCausalLM,
        beta_sampling: float,
        device: torch.device | str = "cpu",
    ):
        super().__init__(model, beta_sampling, device)
        self.batch_preparer = ProGen3BatchPreparer()
        vocab = self.batch_preparer.tokenizer.get_vocab()
        self.pad_token_id = vocab["<pad>"]
        self.eos_token_id = vocab["<eos>"]
        self.nterm_token_id = vocab["1"]
        self.cterm_token_id = vocab["2"]

        self.default_gen_config = GenerationConfig(
            do_sample=True,
            use_cache=True,
            output_logits=True,  #  maybe should use output_scores=True to get processed logits
            return_dict_in_generate=True,
            temperature=1.0 / self.beta_sampling,
            top_p=1.0,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            min_new_tokens=0,
        )

    def _prepare_generation_inputs(
        self, prompt: list[str] | torch.Tensor
    ) -> dict[str, torch.LongTensor]:
        """Prepare inputs for generation from prompt strings or token IDs."""

        if isinstance(prompt, list):
            inputs_ids = [
                self.batch_preparer.tokenizer.encode(f"<bos>1{sequence}").ids
                for sequence in prompt
            ]
            inputs_ids = torch.tensor(inputs_ids, device=self.device, dtype=torch.long)
        elif isinstance(prompt, torch.Tensor):
            inputs_ids = prompt.to(self.device)
        return {
            "input_ids": inputs_ids,
            "position_ids": (
                torch.arange(inputs_ids.size(1), device=self.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(inputs_ids.size(0), -1)
            ),
            "sequence_ids": torch.zeros_like(inputs_ids),
        }

    def _format_generation_output_for_mcmc(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        previous_state: MCMCState | None = None,
    ) -> MCMCState:
        temperature = self.default_gen_config.temperature

        # Get the generated tokens
        generated_tokens = sequences[:, -logits.size(1) :]

        # compute log probabilities (with logits processor, pad_tokens have zero log prob)
        legacy_log_probs = F.log_softmax(logits, dim=-1)
        sampling_log_probs = F.log_softmax(logits / temperature, dim=-1)
        # select indices corresponding to generated tokens
        legacy_log_probs = torch.gather(
            legacy_log_probs, dim=2, index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)
        sampling_log_probs = torch.gather(
            sampling_log_probs, dim=2, index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Set log_probs for padding mask to 0
        padding_mask = generated_tokens == self.pad_token_id
        legacy_log_probs = torch.where(padding_mask, 0, legacy_log_probs)
        sampling_log_probs = torch.where(padding_mask, 0, sampling_log_probs)

        if previous_state is not None:
            # Concatenate with previous log probs
            legacy_log_probs = torch.cat(
                [previous_state.legacy_log_probs, legacy_log_probs], dim=1
            )
            sampling_log_probs = torch.cat(
                [previous_state.sampling_log_probs, sampling_log_probs], dim=1
            )

        return MCMCState(
            sequences=sequences,
            legacy_log_probs=legacy_log_probs,
            sampling_log_probs=sampling_log_probs,
        )

    # @suppress_generation_warnings()
    @torch.inference_mode()
    def generate_block(
        self,
        input_state: list[str] | torch.Tensor | MCMCState,
        block_size: int,
        processors: LogitsProcessorList | None = None,
    ) -> MCMCState:
        # Configure generation settings
        gen_config = deepcopy(self.default_gen_config)
        gen_config.max_new_tokens = block_size

        # Prepare inputs
        is_first_block = not isinstance(input_state, MCMCState)
        sequences = input_state if is_first_block else input_state.sequences
        inputs = self._prepare_generation_inputs(sequences)
        num_input_tokens = inputs["input_ids"].size(1)

        # Initialize finished mask and LogitsProcessors
        mask_finished = (inputs["input_ids"] == self.eos_token_id).any(dim=1)
        if processors is None:
            processors = LogitsProcessorList([])
            
        processors = deepcopy(processors)
        processors.append(
            CTermLogitsProcessor(
                mask_finished, self.pad_token_id, self.eos_token_id, self.cterm_token_id
            )
        )

        # Handle KV Cache for the prompt
        cached_length = num_input_tokens - 1
        key_value_cache: DynamicCache | None = None
        if cached_length > 0:
            cached_encoding = {k: v[:, :cached_length] for k, v in inputs.items()}
            key_value_cache = self.model(
                **cached_encoding, use_cache=True, return_dict=True
            ).past_key_values

        # Single model call for the whole batch
        outputs: GenerateDecoderOnlyOutput = self.model.generate(
            **inputs,
            generation_config=gen_config,
            past_key_values=key_value_cache,
            logits_processor=processors,
        )

        logits = torch.stack(outputs.logits, dim=1)
        state = self._format_generation_output_for_mcmc(
            outputs.sequences, logits, input_state if not is_first_block else None
        )
        return state

    @torch.inference_mode()
    def resample(
        self,
        state: MCMCState,
        window_size: int,
        processors: LogitsProcessorList | None = None,
    ) -> MCMCState:
        """Resample the last k tokens of each sequence using Metropolis-Hastings acceptance."""
        state = state.to(self.device)
        batch_size, seq_len = state.sequences.size()
        original_prompt_len = seq_len - state.sampling_log_probs.size(1)

        # Clone to create new containers
        new_sequences = state.sequences.clone()
        new_per_token_legacy_log_probs = state.legacy_log_probs.clone()
        new_per_token_sampling_log_probs = state.sampling_log_probs.clone()

        finished = (state.sequences == self.eos_token_id).any(dim=1)

        # Calculate actual sequences lengths
        actual_lengths = (state.sequences != self.pad_token_id).sum(dim=1)

        # Pick random depth to resample
        m = torch.randint(1, window_size + 1, (1,), device=self.device).item()
        start_positions = torch.clamp(
            actual_lengths - m, min=original_prompt_len
        )  # original clamp min was 2 (<bos> and "1"), but changed to accomodate longer prompts
        min_start_positions = start_positions.min().item()

        for pos in range(min_start_positions, seq_len):

            # Active mask: in resampling window, not finished, before actual end
            active_mask = (pos >= start_positions) & (pos < actual_lengths) & (~finished)

            if not active_mask.any():
                continue

            input_ids = new_sequences[:, :pos]
            logits = self.model(
                **self._prepare_generation_inputs(input_ids),
                use_cache=False,
                return_dict=True,
            ).logits[:, -1, :]

            # Apply any custom processors
            # logits = processors(input_ids, logits)
            if processors:
                logits = processors(input_ids, logits)

            # Execute mask processor
            mask_finished = (input_ids == self.eos_token_id).any(dim=1)
            mask_processor = CTermLogitsProcessor(
                mask_finished, self.pad_token_id, self.eos_token_id, self.cterm_token_id
            )
            logits = mask_processor(input_ids, logits)

            # Compute probabilities
            probs_sampling = F.softmax(logits * self.beta_sampling, dim=-1)
            probs_legacy = F.softmax(logits, dim=-1)

            candidate_token = torch.multinomial(probs_sampling, num_samples=1).squeeze(1)
            original_token = new_sequences[:, pos]
            candidate_token = torch.where(finished, self.pad_token_id, candidate_token)

            # Update sequences only where active
            new_token = torch.where(active_mask, candidate_token, original_token)
            new_sequences[:, pos] = new_token

            last_log_prob = torch.log(probs_legacy + 1e-20)
            last_sampling_log_prob = torch.log(probs_sampling + 1e-20)

            legacy_log_prob = last_log_prob.gather(
                dim=1, index=candidate_token.unsqueeze(1)
            ).squeeze(1)
            sampling_log_prob = last_sampling_log_prob.gather(
                dim=1, index=candidate_token.unsqueeze(1)
            ).squeeze(1)

            # Zero out log probs for finished sequences
            legacy_log_prob = torch.where(finished, 0, legacy_log_prob)
            sampling_log_prob = torch.where(finished, 0, sampling_log_prob)

            # Update log probs array (offset for prompt length)
            log_prob_idx = pos - original_prompt_len
            if log_prob_idx < new_per_token_legacy_log_probs.size(1):
                new_per_token_legacy_log_probs[:, log_prob_idx] = torch.where(
                    active_mask, legacy_log_prob, new_per_token_legacy_log_probs[:, log_prob_idx]
                )
                new_per_token_sampling_log_probs[:, log_prob_idx] = torch.where(
                    active_mask,
                    sampling_log_prob,
                    new_per_token_sampling_log_probs[:, log_prob_idx],
                )

        return MCMCState(
            new_sequences, new_per_token_legacy_log_probs, new_per_token_sampling_log_probs
        )
