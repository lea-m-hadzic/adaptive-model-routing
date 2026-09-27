"""Belief updates and belief-space value functions.

The belief is a single number b = P(s=1 | history). All functions accept floats or
numpy arrays of beliefs so they can be evaluated on a grid.
"""

from __future__ import annotations

import numpy as np

from .models import RewardModel, ScoreModel, SensorModel


def update(belief, obs, sensor: SensorModel | ScoreModel):
    """Bayes' rule: P(s=1 | o) = p(o|s=1) b / [p(o|s=1) b + p(o|s=0) (1-b)].

    `obs` is a bool for a binary sensor or a real-valued score for a ScoreModel;
    in the latter case p(o|s) is a density, but only the ratio matters.
    """
    like1 = sensor.likelihood(obs, 1)
    like0 = sensor.likelihood(obs, 0)
    numer = like1 * belief
    return numer / (numer + like0 * (1.0 - belief))


def prob_positive(belief, sensor: SensorModel):
    """Predictive probability that the sensor fires: P(o=+ | b)."""
    return sensor.tpr * belief + sensor.fpr * (1.0 - belief)


def accept_value(belief, reward: RewardModel):
    """Expected terminal reward of accepting now: b R_TP - (1-b) C_FP."""
    return belief * reward.r_tp - (1.0 - belief) * reward.c_fp


def reject_value(belief, reward: RewardModel):
    """Expected terminal reward of rejecting now: -b C_FN."""
    return -belief * reward.c_fn


def decide_now_value(belief, reward: RewardModel):
    """Value of stopping and taking the better terminal action."""
    return np.maximum(accept_value(belief, reward), reject_value(belief, reward))


def accept_threshold(reward: RewardModel) -> float:
    """Belief at which accepting and rejecting have equal expected reward.

    Solving b R_TP - (1-b) C_FP = -b C_FN gives b* = C_FP / (R_TP + C_FN + C_FP).
    Note R_TP and C_FN only affect decisions through their sum.
    """
    return reward.c_fp / (reward.r_tp + reward.c_fn + reward.c_fp)


def query_then_decide_value(belief, sensor: SensorModel, reward: RewardModel):
    """Expected value of querying `sensor` once, then deciding (net of query cost).

    sum_o P(o | b) * max_decision( b' = update(b, o) )  -  cost
    """
    p_pos = prob_positive(belief, sensor)
    v_pos = decide_now_value(update(belief, True, sensor), reward)
    v_neg = decide_now_value(update(belief, False, sensor), reward)
    return p_pos * v_pos + (1.0 - p_pos) * v_neg - sensor.cost


def escalation_band(sensor: SensorModel, reward: RewardModel, n_grid: int = 100_001) -> tuple[float, float] | None:
    """Beliefs where querying `sensor` beats deciding immediately.

    The value of information (before cost) is zero at b=0 and b=1 and peaks at the
    accept threshold, so the set {b : query value > decide-now value} is an interval
    [low, high]. We locate it on a fine grid. Returns None if querying never pays.
    """
    b = np.linspace(0.0, 1.0, n_grid)
    gain = query_then_decide_value(b, sensor, reward) - decide_now_value(b, reward)
    worth_it = b[gain > 0]
    if worth_it.size == 0:
        return None
    return float(worth_it.min()), float(worth_it.max())
