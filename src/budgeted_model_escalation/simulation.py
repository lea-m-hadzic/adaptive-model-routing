"""Monte Carlo simulation of policies on synthetic items."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import belief as bel
from .environment import ItemBatch, sample_items, terminal_reward
from .models import Action, ProblemParams
from .metrics import summarize_by_policy
from .policies import CheapOnly, ExpensiveOnly, FixedCascade, Policy, PosteriorThreshold

MAX_STEPS = 10  # safety net; valid policies stop after <= 3 actions
_SENSOR_KEY = {Action.QUERY_CHEAP: "cheap", Action.QUERY_EXPENSIVE: "expensive"}


def run_episode(
    policy: Policy,
    state: bool,
    outcomes: dict[str, float | bool],
    env: ProblemParams,
    agent: ProblemParams,
) -> dict:
    """Run one item through the POMDP loop: act -> observe -> Bayes update.

    `env` holds the true parameters (costs, rewards); `agent` holds the parameters
    the agent *believes* and uses for its belief updates. They are identical unless
    we are studying model misspecification.
    """
    b = agent.p_target
    observed: dict[str, float | bool] = {}
    cost = 0.0
    for _ in range(MAX_STEPS):
        action = policy.act(b, observed)
        if action in (Action.ACCEPT, Action.REJECT):
            accepted = action is Action.ACCEPT
            r = terminal_reward(state, accepted, env.reward) - cost
            return {
                "state": state,
                "accepted": accepted,
                "cost": cost,
                "used_cheap": "cheap" in observed,
                "used_expensive": "expensive" in observed,
                "final_belief": b,
                "reward": r,
            }
        key = _SENSOR_KEY[action]
        if key in observed:
            raise RuntimeError(f"{policy.name} queried the {key} model twice")
        o = outcomes[key]
        observed[key] = o
        cost += env.sensor_for(action).cost  # pay the true cost
        b = bel.update(b, o, agent.sensor_for(action))  # update with the assumed model
    raise RuntimeError(f"{policy.name} did not terminate within {MAX_STEPS} steps")


def run_policy(
    policy: Policy,
    items: ItemBatch,
    env: ProblemParams,
    agent: ProblemParams | None = None,
) -> pd.DataFrame:
    """Per-item results for one policy on a fixed batch of items."""
    agent = agent or env
    rows = [
        run_episode(
            policy,
            bool(items.states[i]),
            {"cheap": float(items.cheap_obs[i]), "expensive": bool(items.expensive_obs[i])},
            env,
            agent,
        )
        for i in range(len(items))
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "item", np.arange(len(items)))
    df.insert(0, "policy", policy.name)
    return df


def simulate(
    policies: list[Policy],
    env: ProblemParams,
    n_items: int,
    seed: int,
    agent: ProblemParams | None = None,
) -> pd.DataFrame:
    """Evaluate all policies on the same sampled items (common random numbers)."""
    items = sample_items(env, n_items, np.random.default_rng(seed))
    return pd.concat([run_policy(p, items, env, agent) for p in policies], ignore_index=True)


# ---------------------------------------------------------------------------
# Tuned cascade baseline
# ---------------------------------------------------------------------------

# Candidate score thresholds; +/-inf give "never escalate" and "always escalate".
CASCADE_GRID = np.concatenate([[-np.inf], np.round(np.arange(-1.0, 4.0001, 0.05), 2), [np.inf]])


def cascade_mean_reward(items: ItemBatch, env: ProblemParams, threshold: float) -> float:
    """Mean net reward of FixedCascade(threshold) on `items`, computed in closed form.

    Vectorized equivalent of running the cascade through `run_episode`; used only
    to make threshold tuning fast. A test checks the two agree.
    """
    r = env.reward
    s = items.states
    escalate = items.cheap_obs >= threshold
    accept = escalate & items.expensive_obs
    terminal = np.where(accept, np.where(s, r.r_tp, -r.c_fp), np.where(s, -r.c_fn, 0.0))
    cost = env.cheap.cost + escalate * env.expensive.cost
    return float((terminal - cost).mean())


def tune_cascade(env: ProblemParams, n_items: int, tune_seed: int, grid=CASCADE_GRID) -> FixedCascade:
    """Pick the cascade threshold that maximizes mean reward on a separate tuning batch.

    This is the model-free baseline: it needs labeled items from the true
    environment (here, a fresh simulated batch) but no observation model.
    Evaluate it on a different seed to avoid hindsight bias.
    """
    items = sample_items(env, n_items, np.random.default_rng(tune_seed))
    rewards = [cascade_mean_reward(items, env, t) for t in grid]
    return FixedCascade(float(grid[int(np.argmax(rewards))]), name="tuned_cascade")


def standard_policies(env: ProblemParams, n_items: int, tune_seed: int) -> list[Policy]:
    """The five policies compared throughout the paper, adaptive policy last."""
    t = env.cheap.decision_threshold
    return [
        CheapOnly(t),
        ExpensiveOnly(),
        FixedCascade(t),
        tune_cascade(env, n_items, tune_seed),
        PosteriorThreshold.from_params(env),
    ]


def compare(env: ProblemParams, n_items: int, seed: int, tune_seed: int) -> pd.DataFrame:
    """Summary of the standard policies, with paired differences vs the tuned cascade."""
    results = simulate(standard_policies(env, n_items, tune_seed), env, n_items, seed)
    return summarize_by_policy(results, reference="tuned_cascade")


def sweep(
    name: str,
    values: list[float],
    make_params: Callable[[float], ProblemParams],
    n_items: int,
    seed: int,
    tune_seed: int,
) -> pd.DataFrame:
    """`compare` at each value of one parameter.

    The adaptive policy's thresholds are re-derived at every point (it plans with
    the correct parameters) and the tuned cascade is re-tuned at every point; the
    fixed baselines keep their rules.
    """
    frames = []
    for v in values:
        summary = compare(make_params(v), n_items, seed, tune_seed)
        summary.insert(0, name, v)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)
