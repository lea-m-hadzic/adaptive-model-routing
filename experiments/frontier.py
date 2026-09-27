"""Budget frontier: terminal reward and recall vs average query cost.

Budgeted problem: maximize E[terminal reward] subject to E[query cost] <= B.
Its Lagrangian is E[terminal reward] - lambda * E[cost]. For each lambda the
adaptive policy plans as if the expensive query cost lambda * c_exp; lambda = 1 is
the net-reward-optimal policy used everywhere else. Sweeping lambda traces the
adaptive frontier; sweeping the score threshold t traces the cascade frontier.

    python experiments/frontier.py

Writes results/frontier.csv and figures/frontier.png.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import base_parser, output_path
from style import LABEL, MUTED, TEXT_WIDTH, line_kw, point_kw, save, top_legend

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


def plot(df: pd.DataFrame, stem) -> list:
    panels = [("terminal_reward", "Terminal reward"), ("recall", "Recall")]
    fig, axes = plt.subplots(len(P_TARGETS), 2, figsize=(TEXT_WIDTH, 4.7), sharex=True)
    letters = iter("abcdefgh")
    for i, p in enumerate(P_TARGETS):
        d = df[df.p_target == p]
        ada = d[d.family == "adaptive"].sort_values("avg_cost")
        cas = d[d.family == "cascade"].sort_values("avg_cost")
        lam1 = ada[np.isclose(ada.knob, 1.0)].iloc[0]
        for j, (col, title) in enumerate(panels):
            ax = axes[i, j]
            if col == "terminal_reward":
                # Iso-net-reward line through the lambda = 1 point: slope 1 in (cost, terminal reward).
                xs = np.array([d.avg_cost.min(), d.avg_cost.max()])
                ax.plot(xs, lam1[col] + (xs - lam1.avg_cost), color=MUTED, linewidth=0.8,
                        linestyle=(0, (1, 2)), zorder=1, label="Equal net reward as $\\lambda = 1$")
            ax.plot(cas.avg_cost, cas[col], **line_kw("tuned_cascade", marker="o", markersize=2.2,
                                                      markeredgewidth=0, linewidth=1.2,
                                                      label="Cascade, threshold $t$ swept"))
            ax.plot(ada.avg_cost, ada[col], **line_kw("posterior_threshold", markersize=2.6, markeredgewidth=0,
                                                      label="Adaptive, $\\lambda$ swept"))
            for fam in ["cheap_only", "expensive_only", "fixed_cascade"]:
                row = d[d.family == fam].iloc[0]
                ax.plot(row.avg_cost, row[col], **point_kw(fam, size=5.5))
            row = d[d.family == "tuned_cascade"].iloc[0]
            ax.plot(row.avg_cost, row[col], **point_kw("tuned_cascade", size=6.5))
            ax.plot(lam1.avg_cost, lam1[col], **dict(point_kw("posterior_threshold", size=7),
                                                     label="Adaptive, $\\lambda = 1$"))
            ax.set_title(f"({next(letters)}) {title}, $p = {p:g}$")
            if i == len(P_TARGETS) - 1:
                ax.set_xlabel("Expected query cost per item")
            ax.set_xlim(0.6, 10.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = ["Adaptive, $\\lambda$ swept", "Adaptive, $\\lambda = 1$", "Cascade, threshold $t$ swept",
             LABEL["tuned_cascade"], LABEL["fixed_cascade"], LABEL["cheap_only"], LABEL["expensive_only"],
             "Equal net reward as $\\lambda = 1$"]
    by_label = dict(zip(labels, handles))
    fig.tight_layout(h_pad=1.2, w_pad=2.0)
    top_legend(fig, [by_label[k] for k in order], order, ncol=4)
    return save(fig, stem)


def run(n_items: int, seed: int, tune_seed: int) -> pd.DataFrame:
    df = pd.concat([frontier_rows(ProblemParams().with_updates(p_target=p), n_items, seed, tune_seed)
                    for p in P_TARGETS], ignore_index=True)
    csv, png = output_path("frontier", "csv"), output_path("frontier", "png")
    df.to_csv(csv, index=False)
    figs = plot(df, png)
    for p in P_TARGETS:
        d = df[df.p_target == p]
        print(f"p={p}: max cascade advantage over adaptive frontier (terminal reward at equal cost) "
              f"= {cascade_advantage(d):+.3f}")
        print(d[d.family != "cascade"][["family", "knob", "avg_cost", "terminal_reward", "recall",
                                         "expected_reward"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {csv.name}, " + ", ".join(f.name for f in figs))
    return df


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
