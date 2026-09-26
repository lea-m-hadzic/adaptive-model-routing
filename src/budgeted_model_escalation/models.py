"""Problem parameters: sensor (classifier) models, reward model, and prior."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np


class Action(Enum):
    QUERY_CHEAP = "query_cheap"
    QUERY_EXPENSIVE = "query_expensive"
    ACCEPT = "accept_target"
    REJECT = "reject_target"


def _normal_sf(x: float) -> float:
    """P(Z > x) for a standard normal Z."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


@dataclass(frozen=True)
class SensorModel:
    """A noisy binary classifier with known observation model.

    tpr = P(o=+ | s=1)   (sensitivity)
    fpr = P(o=+ | s=0)   (1 - specificity)
    """

    name: str
    tpr: float
    fpr: float
    cost: float

    def likelihood(self, positive: bool, state: int) -> float:
        """P(o | s) for this sensor."""
        p_pos = self.tpr if state == 1 else self.fpr
        return p_pos if positive else 1.0 - p_pos

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one observation per item given its hidden state (True = positive)."""
        u = rng.random(len(states))
        return np.where(states, u < self.tpr, u < self.fpr)


@dataclass(frozen=True)
class ScoreModel:
    """A classifier that returns a real-valued confidence score x (think: a logit).

    Class-conditional score distributions (equal-variance Gaussians):
        x | s=0 ~ N(0, 1)
        x | s=1 ~ N(separation, 1)

    `separation` is d', the distance between the class means in noise standard
    deviations; it is the single knob for model quality (ROC AUC = Phi(d' / sqrt 2)).

    `decision_threshold` is a fixed operating point used only by baselines that turn
    the score into a hard label (x >= threshold -> "positive"). The Bayesian agent
    ignores it and uses the full likelihood p(x | s) instead.
    """

    name: str
    separation: float
    cost: float
    decision_threshold: float

    def likelihood(self, score, state: int):
        """Density p(x | s). Works on floats or numpy arrays."""
        mu = self.separation if state == 1 else 0.0
        return np.exp(-0.5 * (score - mu) ** 2) / math.sqrt(2.0 * math.pi)

    def sample(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(np.where(states, self.separation, 0.0), 1.0)

    @property
    def tpr(self) -> float:
        """P(x >= decision_threshold | s=1): recall of the thresholded score."""
        return _normal_sf(self.decision_threshold - self.separation)

    @property
    def fpr(self) -> float:
        """P(x >= decision_threshold | s=0)."""
        return _normal_sf(self.decision_threshold)

    def score_at_posterior(self, prior: float, posterior: float) -> float:
        """Score x that moves belief `prior` to exactly `posterior`.

        For equal-variance Gaussians the log-likelihood ratio is linear in x,
            log p(x|1)/p(x|0) = d' x - d'^2 / 2,
        so posterior log-odds = prior log-odds + d' x - d'^2 / 2. Solving for x turns
        belief thresholds into score thresholds, which makes policies interpretable.
        """
        if posterior <= 0.0:
            return -math.inf
        if posterior >= 1.0:
            return math.inf
        logit = lambda p: math.log(p / (1.0 - p))
        d = self.separation
        return (logit(posterior) - logit(prior)) / d + d / 2.0


@dataclass(frozen=True)
class RewardModel:
    """Terminal rewards for the final accept/reject decision.

    A true negative earns 0. Query costs are charged separately via each sensor's cost.
    """

    r_tp: float = 500.0  # reward for accepting a real target
    c_fp: float = 50.0   # penalty for accepting a non-target
    c_fn: float = 500.0  # penalty for rejecting a real target


@dataclass(frozen=True)
class ProblemParams:
    p_target: float = 0.01
    # Defaults chosen so that, at its operating point, the cheap model matches the
    # earlier binary version: TPR ~= 0.75, FPR ~= 0.10.
    cheap: ScoreModel = field(
        default_factory=lambda: ScoreModel("cheap", separation=1.96, cost=1.0, decision_threshold=1.28)
    )
    expensive: SensorModel = field(default_factory=lambda: SensorModel("expensive", tpr=0.95, fpr=0.02, cost=10.0))
    reward: RewardModel = field(default_factory=RewardModel)

    def sensor_for(self, action: Action) -> ScoreModel | SensorModel:
        if action is Action.QUERY_CHEAP:
            return self.cheap
        if action is Action.QUERY_EXPENSIVE:
            return self.expensive
        raise ValueError(f"{action} is not a query action")

    def with_updates(self, **kwargs) -> "ProblemParams":
        """Convenience for sweeps, e.g. params.with_updates(p_target=0.05)."""
        return replace(self, **kwargs)
