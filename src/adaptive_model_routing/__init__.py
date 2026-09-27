"""Budgeted model escalation for rare-target search."""

from .models import Action, ProblemParams, RewardModel, ScoreModel, SensorModel
from .policies import CheapOnly, ExpensiveOnly, FixedCascade, PosteriorThreshold, default_policies
from .simulation import compare, simulate, standard_policies, sweep, tune_cascade
from .metrics import summarize_by_policy

__all__ = [
    "Action",
    "ProblemParams",
    "RewardModel",
    "ScoreModel",
    "SensorModel",
    "CheapOnly",
    "ExpensiveOnly",
    "FixedCascade",
    "PosteriorThreshold",
    "default_policies",
    "compare",
    "simulate",
    "standard_policies",
    "tune_cascade",
    "sweep",
    "summarize_by_policy",
]
