"""Synthetic rare-target environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ProblemParams, RewardModel


@dataclass
class ItemBatch:
    """Hidden states and the observation each model *would* return for each item.

    Each model is queried at most once per item, and observations are conditionally
    independent given s, so drawing them up front is equivalent to drawing them on
    demand. It also gives every policy the same items and model outputs (common
    random numbers), which makes policy comparisons much less noisy.
    """

    states: np.ndarray         # bool, True = target present
    cheap_obs: np.ndarray      # float, cheap model's confidence score
    expensive_obs: np.ndarray  # bool, True = expensive model says positive

    def __len__(self) -> int:
        return len(self.states)


def sample_items(params: ProblemParams, n_items: int, rng: np.random.Generator) -> ItemBatch:
    states = rng.random(n_items) < params.p_target
    return ItemBatch(
        states=states,
        cheap_obs=params.cheap.sample(states, rng),
        expensive_obs=params.expensive.sample(states, rng),
    )


def terminal_reward(state: bool, accepted: bool, reward: RewardModel) -> float:
    if accepted:
        return reward.r_tp if state else -reward.c_fp
    return -reward.c_fn if state else 0.0
