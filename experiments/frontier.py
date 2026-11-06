"""Budget frontier: terminal reward and recall vs. average query cost.

Budgeted problem: maximize E[terminal reward] subject to E[query cost] <= B.
Its Lagrangian is E[terminal reward] - lambda * E[cost]. For each lambda the
adaptive policy plans as if the expensive query cost lambda * c_exp; lambda = 1 is
the net-reward-optimal policy used everywhere else. Sweeping lambda traces the
adaptive frontier; sweeping the score threshold t traces the cascade frontier.

    python experiments/frontier.py

Writes results/frontier.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import base_parser, csv_path

from adaptive_model_routing import CheapOnly, ExpensiveOnly, FixedCascade, PosteriorThreshold, ProblemParams
from adaptive_model_routing.environment import sample_items
from adaptive_model_routing.metrics import summarize
from adaptive_model_routing.simulation import run_policy, tune_cascade

P_TARGETS = [0.01, 0.1]
LAMBDAS = np.unique(np.concatenate([np.geomspace(0.02, 8, 20), [1.0]]))
CASCADE_T = np.round(np.arange(-1.0, 3.51, 0.25), 2)


def frontier_rows(params: ProblemParams, n_items: int, seed: int, tune_seed: int) -> pd.DataFrame:
    """Evaluate every frontier point on one shared batch of items."""
    items = sample_items(params, n_items, np.random.default_rng(seed))
    t_fix = params.cheap.decision_threshold
    tuned = tune_cascade(params, n_items, tune_seed)
    entries = (
        [("adaptive", lam, PosteriorThreshold.from_params(params, cost_scale=lam)) for lam in LAMBDAS]
        + [("cascade", t, FixedCascade(t)) for t in CASCADE_T]
        + [("tuned_cascade", tuned.score_threshold, tuned),
           ("fixed_cascade", t_fix, FixedCascade(t_fix)),
           ("cheap_only", np.nan, CheapOnly(t_fix)),
           ("expensive_only", np.nan, ExpensiveOnly())]
    )
    rows = []
    for family, knob, policy in entries:
        # Summarize one policy at a time to keep memory flat.
        rows.append({"p_target": params.p_target, "family": family, "knob": knob,
                     **summarize(run_policy(policy, items, params))})
    return pd.DataFrame(rows)


def cascade_advantage(df: pd.DataFrame) -> float:
    """Largest amount by which any cascade point beats the adaptive frontier at equal cost.

    The adaptive frontier is interpolated linearly in cost. Positive values would
    mean the cascade is above the adaptive frontier somewhere.
    """
    a = df[df.family == "adaptive"].sort_values("avg_cost")
    c = df[df.family == "cascade"]
    inside = c[(c.avg_cost >= a.avg_cost.min()) & (c.avg_cost <= a.avg_cost.max())]
    interp = np.interp(inside.avg_cost, a.avg_cost, a.terminal_reward)
    return float(np.max(inside.terminal_reward.to_numpy() - interp))


def run(n_items: int, seed: int, tune_seed: int) -> pd.DataFrame:
    df = pd.concat([frontier_rows(ProblemParams().with_updates(p_target=p), n_items, seed, tune_seed)
                    for p in P_TARGETS], ignore_index=True)
    csv = csv_path("frontier")
    df.to_csv(csv, index=False)
    for p in P_TARGETS:
        d = df[df.p_target == p]
        print(f"p={p}: max cascade advantage over adaptive frontier (terminal reward at equal cost) "
              f"= {cascade_advantage(d):+.3f}")
        print(d[d.family != "cascade"][["family", "knob", "avg_cost", "terminal_reward", "recall",
                                         "expected_reward"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote results/{csv.name}")
    return df


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
