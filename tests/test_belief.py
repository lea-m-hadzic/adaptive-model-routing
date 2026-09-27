import numpy as np
import pytest

from adaptive_model_routing import belief as bel
from adaptive_model_routing.models import RewardModel, ScoreModel, SensorModel

CHEAP = SensorModel("cheap", tpr=0.75, fpr=0.10, cost=1.0)  # binary sensor, for hand-checkable math
SCORE = ScoreModel("cheap", separation=2.0, cost=1.0, decision_threshold=1.28)
EXPENSIVE = SensorModel("expensive", tpr=0.95, fpr=0.02, cost=10.0)
REWARD = RewardModel(r_tp=500, c_fp=50, c_fn=500)


def test_update_matches_hand_computation():
    # P(s=1 | +) = 0.75*0.01 / (0.75*0.01 + 0.10*0.99)
    assert bel.update(0.01, True, CHEAP) == pytest.approx(0.0075 / 0.1065)
    # P(s=1 | -) = 0.25*0.01 / (0.25*0.01 + 0.90*0.99)
    assert bel.update(0.01, False, CHEAP) == pytest.approx(0.0025 / 0.8935)


def test_positive_raises_and_negative_lowers_belief():
    assert bel.update(0.01, True, CHEAP) > 0.01 > bel.update(0.01, False, CHEAP)


def test_uninformative_sensor_leaves_belief_unchanged():
    coin = SensorModel("coin", tpr=0.3, fpr=0.3, cost=0.0)
    assert bel.update(0.2, True, coin) == pytest.approx(0.2)
    assert bel.update(0.2, False, coin) == pytest.approx(0.2)


def test_update_order_does_not_matter():
    a = bel.update(bel.update(0.01, True, CHEAP), False, EXPENSIVE)
    b = bel.update(bel.update(0.01, False, EXPENSIVE), True, CHEAP)
    assert a == pytest.approx(b)


def test_belief_is_a_martingale():
    # E[b' | b] = b: expected posterior equals the prior.
    b = 0.03
    p = bel.prob_positive(b, CHEAP)
    expected = p * bel.update(b, True, CHEAP) + (1 - p) * bel.update(b, False, CHEAP)
    assert expected == pytest.approx(b)


def test_accept_threshold_equalizes_terminal_values():
    tau = bel.accept_threshold(REWARD)
    assert bel.accept_value(tau, REWARD) == pytest.approx(bel.reject_value(tau, REWARD))
    assert bel.accept_value(tau + 0.01, REWARD) > bel.reject_value(tau + 0.01, REWARD)


def test_information_is_never_harmful_before_cost():
    free = SensorModel("free", tpr=0.95, fpr=0.02, cost=0.0)
    b = np.linspace(0, 1, 501)
    gain = bel.query_then_decide_value(b, free, REWARD) - bel.decide_now_value(b, REWARD)
    assert np.all(gain >= -1e-9)


def test_escalation_band_contains_threshold_and_vanishes_when_too_costly():
    low, high = bel.escalation_band(EXPENSIVE, REWARD)
    assert low < bel.accept_threshold(REWARD) < high
    pricey = SensorModel("expensive", tpr=0.95, fpr=0.02, cost=1e6)
    assert bel.escalation_band(pricey, REWARD) is None


def test_score_update_matches_closed_form_log_odds():
    # Equal-variance Gaussians: posterior log-odds = prior log-odds + d' x - d'^2 / 2
    prior, x, d = 0.01, 1.5, SCORE.separation
    expected = 1 / (1 + np.exp(-(np.log(prior / (1 - prior)) + d * x - d * d / 2)))
    assert bel.update(prior, x, SCORE) == pytest.approx(expected)


def test_score_posterior_is_monotone_in_score():
    scores = np.linspace(-4, 6, 201)
    post = bel.update(0.01, scores, SCORE)
    assert np.all(np.diff(post) > 0)


def test_score_at_posterior_inverts_update():
    for target in [0.001, 0.05, 0.5, 0.9]:
        x = SCORE.score_at_posterior(0.01, target)
        assert bel.update(0.01, x, SCORE) == pytest.approx(target)


def test_score_model_operating_point_rates():
    # Threshold at 1.28 on N(0,1) gives FPR ~= 0.10; TPR = P(N(2,1) >= 1.28) ~= 0.764
    assert SCORE.fpr == pytest.approx(0.1003, abs=1e-3)
    assert SCORE.tpr == pytest.approx(0.7642, abs=1e-3)


def test_score_model_is_martingale():
    # E[b'] = b under the model's own predictive distribution of scores.
    rng = np.random.default_rng(0)
    b = 0.2
    states = rng.random(400_000) < b
    post = bel.update(b, SCORE.sample(states, rng), SCORE)
    assert post.mean() == pytest.approx(b, abs=2e-3)
