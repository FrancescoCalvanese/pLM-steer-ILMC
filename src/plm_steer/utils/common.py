import argparse

import numpy as np
import random
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Input protein sequence in single-letter amino acid code.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="mcmc_samples",
        help="Directory to save the sampled sequences in FASTA format.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Interval (in MCMC steps) to save intermediate sampled sequences.",
    )
    parser.add_argument(
        "--potentials-config",
        type=str,
        default=None,
        help="Path to the YAML configuration file for potentials.",
    )
    parser.add_argument(
        "--num-sequences", "-n", type=int, default=128, help="Number of sequences to sample."
    )
    parser.add_argument(
        "--beta", "-t", type=float, default=1.0, help="Inverse temperature for MCMC sampling."
    )
    parser.add_argument(
        "--num-mcmc-steps", "-m", type=int, default=100, help="Number of MCMC sampling steps."
    )
    parser.add_argument(
        "--num-steps", "-s", type=int, default=100, help="Number of generation steps."
    )
    parser.add_argument(
        "--block-size", "-b", type=int, default=5, help="Block size for block MCMC sampling."
    )
    parser.add_argument(
        "--resampling-window", "-k", type=int, default=5, help="Resampling window size."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable the progress bar during sampling.",
    )
    return parser
