"""Define custom steering potentials"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from Bio.Align import PairwiseAligner

_DEFAULT_THERMOSTABILITY_ER = {
    # Group 1: stronglyenriched (ratio > 1.10)
    "E": 1.28,
    "K": 1.27,
    "W": 1.24,
    "R": 1.23,
    "Y": 1.22,
    "I": 1.21,
    "F": 1.21,
    "L": 1.14,
    "V": 1.13,
    "P": 1.13,
    # group 2: moderately enriched/depleted (0.90 <= ratio <= 1.10)
    "G": 1.04,
    "C": 0.94,
    "M": 0.91,
    # group 3: strongly depleted (ratio < 0.90)
    "D": 0.82,
    "S": 0.79,
    "N": 0.71,
    "T": 0.71,
    "A": 0.66,
    "H": 0.62,
    "Q": 0.55,
}


class BasePotential(ABC):
    def __init__(self, weight: float, transform: Callable[[torch.Tensor], Any] | None = None):
        self.weight = weight
        self.transform = transform

    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        """Compute and return the potential score as a float."""
        pass


class PotentialRegistry:
    """Central registry to map strings to Potential classes."""

    _registry: dict[str, type[BasePotential]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new potential type."""

        def wrapper(wrapped_class: type[BasePotential]):
            cls._registry[name.lower()] = wrapped_class
            return wrapped_class

        return wrapper

    @classmethod
    def get_potential_class(cls, name: str) -> type[BasePotential]:
        if name.lower() not in cls._registry:
            raise ValueError(
                f"Unknown potential type '{name}'. Available types: {list(cls._registry.keys())}"
            )
        return cls._registry[name.lower()]


@PotentialRegistry.register("neighbouring")
class NeighbouringPotential(BasePotential):
    def __init__(
        self,
        weight: float,
        target: str,
        mismatch_score: float = -1.0,
        gap_score: float = -1.0,
        transform: Callable[[Any], str] | None = None,
    ):
        super().__init__(weight, transform)
        aligner = PairwiseAligner()
        aligner.mode = "global"  # do not trim ends of target
        aligner.match_score = 0.0  # Levenshtein: Match cost is 0
        aligner.mismatch_score = mismatch_score  # Levenshtein: Substitution cost is 1

        # All gaps (Internal & Left) must cost 1.0
        aligner.gap_score = gap_score
        aligner.open_right_deletion_score = 0.0
        aligner.open_right_insertion_score = 0.0
        aligner.extend_right_deletion_score = 0.0
        aligner.extend_right_insertion_score = 0.0

        self.aligner = aligner
        self.target = target

    def __call__(self, query: Any, *args, **kwargs) -> float:
        if self.transform is not None:
            query = self.transform(query)
        if not isinstance(query, str):
            raise ValueError("NeighbouringPotential expects query to be a string.")
        if not query:
            return 0.0
        score = self.aligner.align(self.target, query).score
        return -self.weight * score


@PotentialRegistry.register("thermostability")
class ThermostabilityPotential(BasePotential):
    def __init__(
        self,
        weight: float,
        transform: Callable[[Any], str] | None = None,
        per_residue_score: dict[str, float] | None = None,
    ):
        super().__init__(weight, transform)
        self.per_residue_score = per_residue_score or {
            aa: score - 1 for aa, score in _DEFAULT_THERMOSTABILITY_ER.items()
        }

    def __call__(self, query: Any, *args, **kwargs) -> float:
        if self.transform is not None:
            query = self.transform(query)
        if not isinstance(query, str):
            raise ValueError("ThermostabilityPotential expects query to be a string.")
        stability_score = sum(self.per_residue_score.get(aa, 0.0) for aa in query)
        return -self.weight * stability_score


class CompositePotential:
    def __init__(self, potentials: list[BasePotential]):
        self.potentials = potentials

    def __call__(self, *args, **kwargs) -> float:
        return sum(p(*args, **kwargs) for p in self.potentials)


def load_potentials_from_config(
    config_file: str | Path, transforms: dict[str, Callable]
) -> CompositePotential:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    instantiated_potentials = []

    for pot_cfg in config.get("potentials", []):
        # Extract metadata
        pot_type = pot_cfg.pop("type")
        weight = pot_cfg.pop("weight", 1.0)

        # Get class from registry and build it
        pot_class = PotentialRegistry.get_potential_class(pot_type)

        # Instantiate with remaining config keys
        potential = pot_class(
            weight=weight,
            transform=transforms.get(pot_type),
            **pot_cfg,  # Pass remaining args (target, gap_score, etc.)
        )
        instantiated_potentials.append(potential)

    return CompositePotential(instantiated_potentials)
