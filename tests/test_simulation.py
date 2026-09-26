from dataclasses import replace

import numpy as np
import pytest

from budgeted_model_escalation import FixedCascade, PosteriorThreshold, ProblemParams, compare, simulate
from budgeted_model_escalation.environment import sample_items
from budgeted_model_escalation.metrics import paired_difference, summarize_by_policy
from budgeted_model_escalation.simulation import cascade_mean_reward, run_policy, tune_cascade

PARAMS = ProblemParams().with_updates(p_target=0.05)


def test_vectorized_cascade_reward_matches_simulator():
    items = sample_items(PARAMS, 5_000, np.random.default_rng(0))
    for t in [-np.inf, 0.5, 1.28, 2.5, np.inf]:
        df = run_policy(FixedCascade(t), items, PARAMS)
        assert cascade_mean_reward(items, PARAMS, t) == pytest.approx(df["reward"].mean())


def test_tuned_cascade_is_best_on_its_tuning_batch():
    tuned = tune_cascade(PARAMS, 20_000, tune_seed=5)
    items = sample_items(PARAMS, 20_000, np.random.default_rng(5))
    best = cascade_mean_reward(items, PARAMS, tuned.score_threshold)
    for t in [0.0, 1.0, 1.28, 2.0]:
        assert best >= cascade_mean_reward(items, PARAMS, t)
    assert tuned.name == "tuned_cascade"


def test_paired_difference():
    results = simulate([FixedCascade(1.0, name="a"), FixedCascade(1.5, name="b")], PARAMS, 5_000, seed=0)
    assert paired_difference(results, "a", "a") == (0.0, 0.0)
    mean, se = paired_difference(results, "a", "b")
    ra = results[results.policy == "a"].reward.to_numpy()
    rb = results[results.policy == "b"].reward.to_numpy()
    assert mean == pytest.approx(ra.mean() - rb.mean())
    assert 0 < se < results.reward.std() / np.sqrt(5_000)  # pairing reduces variance


def test_terminal_reward_is_net_reward_plus_cost():
    s = summarize_by_policy(simulate([FixedCascade(1.0)], PARAMS, 5_000, seed=0)).iloc[0]
    assert s["terminal_reward"] == pytest.approx(s["expected_reward"] + s["avg_cost"])


def test_compare_reports_diffs_against_tuned_cascade():
    s = compare(PARAMS, 5_000, seed=0, tune_seed=1).set_index("policy")
    assert list(s.index) == ["cheap_only", "expensive_only", "fixed_cascade", "tuned_cascade", "posterior_threshold"]
    assert s.loc["tuned_cascade", "diff_vs_ref"] == 0.0


def test_cost_scale_shrinks_escalation_band():
    bands = [PosteriorThreshold.from_params(PARAMS, cost_scale=lam) for lam in [0.1, 1.0, 3.0]]
    for looser, tighter in zip(bands, bands[1:]):
        assert looser.low <= tighter.low and looser.high >= tighter.high
    never = PosteriorThreshold.from_params(PARAMS, cost_scale=1e6)
    assert never.low == never.high == never.accept_threshold


def test_correct_assumptions_give_zero_regret():
    items = sample_items(PARAMS, 5_000, np.random.default_rng(0))
    a = run_policy(PosteriorThreshold.from_params(PARAMS), items, PARAMS, agent=PARAMS)
    b = run_policy(PosteriorThreshold.from_params(PARAMS), items, PARAMS)
    assert np.array_equal(a["reward"].to_numpy(), b["reward"].to_numpy())


def test_misspecified_agent_uses_assumed_model_but_pays_true_costs():
    wrong = PARAMS.with_updates(cheap=replace(PARAMS.cheap, separation=4.0))
    items = sample_items(PARAMS, 2_000, np.random.default_rng(0))
    df = run_policy(PosteriorThreshold.from_params(wrong), items, PARAMS, agent=wrong)
    expected_cost = PARAMS.cheap.cost + df["used_expensive"] * PARAMS.expensive.cost
    assert np.allclose(df["cost"], expected_cost)
