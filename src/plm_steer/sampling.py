"""MCMC sampling utilities for sequence generation with custom potentials."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm.auto import trange

from plm_steer.potentials import CompositePotential


@dataclass
class MCMCConfig:
    """Configuration for MCMC sequence generation."""

    block_size: int
    resampling_window: int
    num_iters: int
    num_resamples: int

    def __post_init__(self):
        # No resampling if window is zero, falls back to direct generation
        if self.resampling_window == 0:
            self.num_resamples = 0


@dataclass
class MCMCState:
    """Container for the current state of Markov chains."""

    sequences: torch.LongTensor
    legacy_log_probs: torch.Tensor
    sampling_log_probs: torch.Tensor
    potentials: torch.Tensor | None = None

    def __post_init__(self):
        if self.potentials is None:
            self.potentials = torch.zeros(self.sequences.size(0), device=self.sequences.device)

    def _apply(self, fn):
        """Helper to apply a function to all tensor attributes."""
        return replace(
            self,
            **{k: fn(v) for k, v in self.__dict__.items()},
        )

    def to(self, *args, **kwargs):
        return self._apply(lambda t: t.to(*args, **kwargs))


@dataclass
class LoggingConfig:
    output_dir: str
    save_interval: int
    decode_fn: Callable[[torch.LongTensor], list[str]]
    log_interval: int = 10
    disable_progress_bar: bool = False


class BaseSampler(ABC):
    """Base class for MCMC sequence generation on custom architectures."""

    def __init__(self, model: nn.Module, beta_sampling: float, device: torch.device | str = "cpu"):
        self.model = model.eval().to(device)
        self.beta_sampling = beta_sampling
        self.device = device

    def metropolis_step(
        self,
        current: MCMCState,
        proposal: MCMCState,
        energy_fn: Callable[[MCMCState], torch.Tensor],
    ) -> MCMCState:
        """Standard Metropolis-Hastings acceptance step."""

        e_curr = energy_fn(current)
        e_prop = energy_fn(proposal)

        log_acc_ratio = -(e_prop - e_curr)
        acc_prob = torch.exp(torch.clamp(log_acc_ratio, max=0.0))

        accept_mask = torch.rand(acc_prob.size(), device=self.device) < acc_prob

        # Select accepted states
        new_seq = torch.where(accept_mask.unsqueeze(1), proposal.sequences, current.sequences)
        new_leg = torch.where(
            accept_mask.unsqueeze(1), proposal.legacy_log_probs, current.legacy_log_probs
        )
        new_samp = torch.where(
            accept_mask.unsqueeze(1), proposal.sampling_log_probs, current.sampling_log_probs
        )
        new_pot = torch.where(accept_mask, proposal.potentials, current.potentials)

        return MCMCState(new_seq, new_leg, new_samp, new_pot)

    # To be implemented by subclasses
    @abstractmethod
    def generate_block(
        self, input_state: torch.LongTensor | MCMCState, block_size: int, *args, **kwargs
    ) -> MCMCState:
        pass

    @abstractmethod
    def resample(self, state: MCMCState, window_size: int, *args, **kwargs) -> MCMCState:
        pass


def save_sequences(
    filename: str,
    state: MCMCState | torch.LongTensor,
    decode_fn: Callable[[torch.LongTensor], list[str]],
) -> None:
    is_state_input = isinstance(state, MCMCState)
    sequences = state.sequences if is_state_input else state
    sequence_strings = decode_fn(sequences)

    if is_state_input:
        sum_probs = -state.legacy_log_probs.sum(dim=1)
        lengths = state.legacy_log_probs.count_nonzero(dim=1).clamp(min=1)
        avg_log_probs = (sum_probs / lengths).tolist()

        # Batch string construction for efficiency
        lines = [
            f">sequence_{i} | -ptLogP={prob:.3f}\n{seq}\n"
            for i, (seq, prob) in enumerate(zip(sequence_strings, avg_log_probs))
        ]
    else:
        lines = [f">sequence_{i}\n{seq}\n" for i, seq in enumerate(sequence_strings)]

    with open(filename, "w") as f:
        f.writelines(lines)


def run_block_mcmc_pipeline(
    input_ids: torch.Tensor,
    sampler: BaseSampler,
    sampling_config: MCMCConfig,
    logging_config: LoggingConfig,
    potentials: CompositePotential | None = None,
    generation_kwargs: dict[str, Any] | None = None,
):

    def compute_potentials(state: MCMCState) -> MCMCState:
        if potentials is None:
            return state
        scores = torch.zeros(state.sequences.size(0), device=sampler.device)
        for i in range(len(scores)):
            scores[i] = potentials(query=state.sequences[i])
        return replace(state, potentials=scores)

    def energy_fn(state: MCMCState) -> torch.Tensor:
        log_p = state.legacy_log_probs.sum(dim=1)
        log_q = state.sampling_log_probs.sum(dim=1)
        return -sampler.beta_sampling * log_p + log_q + state.potentials

    # Initialize
    generation_kwargs = generation_kwargs or {}
    output_dir = Path(logging_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = input_ids

    if sampling_config.num_iters == 1 and sampling_config.num_resamples == 0:
        print("Performing direct generation without MCMC...")

    generation_pbar = trange(
        sampling_config.num_iters, desc="Generation", disable=logging_config.disable_progress_bar
    )
    for i in generation_pbar:
        resampling_pbar = trange(
            sampling_config.num_resamples,
            leave=False,
            desc=f"Resampling iter {i+1}",
            disable=logging_config.disable_progress_bar or sampling_config.num_resamples == 0,
        )

        state = sampler.generate_block(
            state,
            sampling_config.block_size,
            **generation_kwargs,
        )
        state = compute_potentials(state)

        for j in resampling_pbar:
            proposal = sampler.resample(
                state, sampling_config.resampling_window, **generation_kwargs
            )
            proposal = compute_potentials(proposal)
            state = sampler.metropolis_step(state, proposal, energy_fn)

            # Update the stats on the side of the inner progress bar
            if (j + 1) % logging_config.log_interval == 0:
                avg_energy = -state.legacy_log_probs.sum(dim=1).mean().item()
                if not logging_config.disable_progress_bar:
                    resampling_pbar.set_postfix({"Avg E": f"{avg_energy:.4f}"})
                else:
                    print(
                        f"Block generation step {i+1}, resampling iteration {j+1}: Avg E = {avg_energy:.4f}",
                        flush=True,
                    )

        # Logging
        if (i + 1) % logging_config.save_interval == 0:
            save_path = output_dir / f"step_{i+1}_sequences.fasta"
            save_sequences(save_path, state, logging_config.decode_fn)

        torch.cuda.empty_cache()

    # Always save final sequences
    save_path = output_dir / "final_sequences.fasta"
    save_sequences(save_path, state, logging_config.decode_fn)

    return state
