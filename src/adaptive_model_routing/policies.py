"""Policies map (belief, observations so far) -> action.

`observed` maps a sensor name to its observation: "cheap" -> real-valued score,
"expensive" -> bool (True = positive). Each sensor is queried at most once per item.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from . import belief as bel
from .models import Action, ProblemParams


class Policy(Protocol):
    name: str

    def act(self, belief: float, observed: dict[str, float | bool]) -> Action: ...


def _decide(positive: bool) -> Action:
    return Action.ACCEPT if positive else Action.REJECT


class CheapOnly:
    """Query the cheap model once; accept iff its score clears a fixed threshold."""

    name = "cheap_only"

    def __init__(self, score_threshold: float):
        self.score_threshold = score_threshold

    def act(self, belief: float, observed: dict[str, float | bool]) -> Action:
        if "cheap" not in observed:
            return Action.QUERY_CHEAP
        return _decide(observed["cheap"] >= self.score_threshold)


class ExpensiveOnly:
    """Query the expensive model once and take its answer at face value."""

    name = "expensive_only"

    def act(self, belief: float, observed: dict[str, float | bool]) -> Action:
        if "expensive" not in observed:
            return Action.QUERY_EXPENSIVE
        return _decide(observed["expensive"])


class FixedCascade:
    """Cheap first; escalate if its score clears a fixed threshold, else reject.

    The final answer is the expensive model's label.
    """

    name = "fixed_cascade"

    def __init__(self, score_threshold: float, name: str | None = None):
        self.score_threshold = score_threshold
        if name is not None:
            self.name = name

    def act(self, belief: float, observed: dict[str, float | bool]) -> Action:
        if "cheap" not in observed:
            return Action.QUERY_CHEAP
        if observed["cheap"] < self.score_threshold:
            return Action.REJECT
        if "expensive" not in observed:
            return Action.QUERY_EXPENSIVE
        return _decide(observed["expensive"])


class PosteriorThreshold:
    """Cheap first, then act on the posterior belief.

    Because the cheap model returns a continuous score, the posterior after it is
    continuous too, so each item lands at its own point relative to the thresholds.
    After the cheap observation:
        b <= low   -> reject  (confident no target)
        b >= high  -> accept  (confident target)
        otherwise  -> query expensive model
    After the expensive observation, accept iff b >= accept_threshold, the
    belief at which accepting and rejecting have equal expected reward.
    """

    name = "posterior_threshold"

    def __init__(self, low: float, high: float, accept_threshold: float, name: str | None = None):
        if not low <= accept_threshold <= high:
            raise ValueError("expected low <= accept_threshold <= high")
        self.low, self.high, self.accept_threshold = low, high, accept_threshold
        if name is not None:
            self.name = name

    @classmethod
    def from_params(cls, params: ProblemParams, cost_scale: float = 1.0, name: str | None = None) -> "PosteriorThreshold":
        """Derive thresholds from the reward model instead of hand-tuning them.

        Escalate exactly when the one-step value of information from the expensive
        model exceeds its cost. Because no further queries are possible after the
        expensive model, this one-step lookahead is Bayes-optimal among
        cheap-first policies (given the assumed params).

        `cost_scale` is a Lagrange multiplier lambda on query cost: the policy plans
        as if the expensive query cost lambda * c_exp. lambda = 1 maximizes net reward;
        sweeping lambda traces the reward-vs-budget frontier (see README).
        """
        tau = bel.accept_threshold(params.reward)
        planning_sensor = replace(params.expensive, cost=cost_scale * params.expensive.cost)
        band = bel.escalation_band(planning_sensor, params.reward)
        if band is None:
            # Escalation never pays: collapse the band so we always decide at tau.
            return cls(low=tau, high=tau, accept_threshold=tau, name=name)
        # min/max guard against grid resolution when the band is extremely narrow.
        return cls(low=min(band[0], tau), high=max(band[1], tau), accept_threshold=tau, name=name)

    def act(self, belief: float, observed: dict[str, float | bool]) -> Action:
        if "cheap" not in observed:
            return Action.QUERY_CHEAP
        if "expensive" in observed:
            return Action.ACCEPT if belief >= self.accept_threshold else Action.REJECT
        if belief <= self.low:
            return Action.REJECT
        if belief >= self.high:
            return Action.ACCEPT
        return Action.QUERY_EXPENSIVE


def default_policies(params: ProblemParams) -> list[Policy]:
    t = params.cheap.decision_threshold
    return [CheapOnly(t), ExpensiveOnly(), FixedCascade(t), PosteriorThreshold.from_params(params)]
