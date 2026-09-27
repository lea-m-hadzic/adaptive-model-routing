from dataclasses import replace

import numpy as np
import pytest

from adaptive_model_routing import (
    Action,
    CheapOnly,
    ExpensiveOnly,
    FixedCascade,
    PosteriorThreshold,
    ProblemParams,
    default_policies,
    simulate,
    summarize_by_policy,
)
from adaptive_model_routing.environment import ItemBatch
from adaptive_model_routing.simulation import run_policy

PARAMS = ProblemParams()
T = PARAMS.cheap.decision_threshold


def test_cheap_only():
    p = CheapOnly(score_threshold=T)
    assert p.act(0.01, {}) is Action.QUERY_CHEAP
    assert p.act(0.5, {"cheap": T + 0.1}) is Action.ACCEPT
    assert p.act(0.5, {"cheap": T - 0.1}) is Action.REJECT


def test_expensive_only():
    p = ExpensiveOnly()
    assert p.act(0.01, {}) is Action.QUERY_EXPENSIVE
    assert p.act(0.5, {"expensive": True}) is Action.ACCEPT


def test_fixed_cascade():
    p = FixedCascade(score_threshold=T)
    assert p.act(0.01, {}) is Action.QUERY_CHEAP
    assert p.act(0.01, {"cheap": T - 0.1}) is Action.REJECT
    assert p.act(0.07, {"cheap": T + 0.1}) is Action.QUERY_EXPENSIVE
    assert p.act(0.8, {"cheap": T + 0.1, "expensive": True}) is Action.ACCEPT
    assert p.act(0.01, {"cheap": T + 0.1, "expensive": False}) is Action.REJECT


def test_posterior_threshold_regions():
    p = PosteriorThreshold(low=0.01, high=0.5, accept_threshold=0.05)
    assert p.act(0.01, {}) is Action.QUERY_CHEAP
    # The policy only looks at the belief; the score value is irrelevant here.
    assert p.act(0.005, {"cheap": 0.0}) is Action.REJECT
    assert p.act(0.9, {"cheap": 0.0}) is Action.ACCEPT
    assert p.act(0.1, {"cheap": 0.0}) is Action.QUERY_EXPENSIVE
    assert p.act(0.06, {"cheap": 0.0, "expensive": True}) is Action.ACCEPT
    assert p.act(0.04, {"cheap": 0.0, "expensive": True}) is Action.REJECT


def test_posterior_threshold_escalates_intermediate_scores_only():
    # Default params: very low scores are rejected, very high accepted, middle escalated.
    pol = PosteriorThreshold.from_params(PARAMS)
    items = ItemBatch(
        states=np.array([False, False, False]),
        cheap_obs=np.array([-2.0, 2.5, 8.0]),
        expensive_obs=np.array([False, False, False]),
    )
    df = run_policy(pol, items, PARAMS)
    assert df["used_expensive"].tolist() == [False, True, False]
    assert df["accepted"].tolist() == [False, False, True]


def test_posterior_threshold_never_escalates_if_expensive_model_is_too_costly():
    params = PARAMS.with_updates(expensive=replace(PARAMS.expensive, cost=1e6))
    items = ItemBatch(np.array([True, False]), np.array([2.5, 2.5]), np.array([True, True]))
    df = run_policy(PosteriorThreshold.from_params(params), items, params)
    assert not df["used_expensive"].any()


def test_cascade_cost_accounting():
    items = ItemBatch(
        states=np.array([True, False, False]),
        cheap_obs=np.array([T + 1, T + 1, T - 1]),
        expensive_obs=np.array([True, False, True]),
    )
    df = run_policy(FixedCascade(T), items, PARAMS)
    assert df["cost"].tolist() == [11.0, 11.0, 1.0]
    assert df["accepted"].tolist() == [True, False, False]
    r = PARAMS.reward
    assert df["reward"].tolist() == [r.r_tp - 11.0, -11.0, -1.0]


def test_monte_carlo_matches_sensor_rates():
    params = PARAMS.with_updates(p_target=0.2)
    s = summarize_by_policy(simulate([CheapOnly(T), FixedCascade(T)], params, n_items=50_000, seed=1))
    cheap, cascade = s.iloc[0], s.iloc[1]
    tpr, fpr = params.cheap.tpr, params.cheap.fpr
    assert cheap["recall"] == pytest.approx(tpr, abs=0.02)
    assert cheap["false_positive_rate"] == pytest.approx(fpr, abs=0.01)
    # Cascade escalates exactly the items whose score clears the threshold.
    assert cascade["frac_expensive"] == pytest.approx(0.2 * tpr + 0.8 * fpr, abs=0.01)
    assert cascade["recall"] == pytest.approx(tpr * params.expensive.tpr, abs=0.02)


def test_simulation_is_reproducible():
    a = summarize_by_policy(simulate(default_policies(PARAMS), PARAMS, n_items=5_000, seed=7))
    b = summarize_by_policy(simulate(default_policies(PARAMS), PARAMS, n_items=5_000, seed=7))
    assert a.equals(b)


def test_adaptive_policy_is_at_least_as_good_as_cheap_first_baselines():
    # With correct beliefs, the VOI-derived policy is optimal among cheap-first
    # policies, so on common random numbers it should not lose to them.
    for p_target in [0.005, 0.05, 0.1]:
        params = PARAMS.with_updates(p_target=p_target)
        s = summarize_by_policy(simulate(default_policies(params), params, n_items=50_000, seed=3))
        r = s.set_index("policy")["expected_reward"]
        tol = 3 * s.set_index("policy")["reward_stderr"].max()
        assert r["posterior_threshold"] >= r["cheap_only"] - tol
        assert r["posterior_threshold"] >= r["fixed_cascade"] - tol
