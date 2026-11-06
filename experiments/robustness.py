"""Robustness to wrong assumptions: the agent plans with incorrect parameters.

The world always uses the true parameters. The adaptive agent builds its thresholds
and updates its belief with *assumed* parameters, off by a ratio r:
    prevalence:  assumed p_target = r * true p_target
    cheap model: assumed d'       = r * true d'   (r > 1: overconfident cheap scores)

For each r we report, on the same items:
    regret        = reward(correctly specified adaptive) - reward(misspecified adaptive)
    diff vs tuned = reward(misspecified adaptive) - reward(tuned cascade)
The tuned cascade is model-free (threshold tuned on labeled data from the true
world), so it is unaffected by misspecification. The break-even ratios are where
"diff vs tuned" crosses zero.

    python experiments/robustness.py

Writes results/robustness.csv and results/robustness_breakeven.csv.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from common import Z95, base_parser, csv_path

from adaptive_model_routing import FixedCascade, PosteriorThreshold, ProblemParams
from adaptive_model_routing.environment import sample_items
from adaptive_model_routing.simulation import run_policy, tune_cascade

TRUE_P = [0.01, 0.1]
RATIOS = {
    "prevalence": np.geomspace(0.1, 10, 11),
    "cheap_dprime": np.geomspace(0.5, 2, 11),
}


def assumed(env: ProblemParams, axis: str, r: float) -> ProblemParams | None:
    if axis == "prevalence":
        p = r * env.p_target
        return env.with_updates(p_target=p) if p < 0.5 else None
    return env.with_updates(cheap=replace(env.cheap, separation=r * env.cheap.separation))


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def robustness_rows(env: ProblemParams, n_items: int, seed: int, tune_seed: int) -> pd.DataFrame:
    items = sample_items(env, n_items, np.random.default_rng(seed))
    reward = lambda pol, agent=None: run_policy(pol, items, env, agent)["reward"].to_numpy()
    r_correct = reward(PosteriorThreshold.from_params(env))
    r_tuned = reward(tune_cascade(env, n_items, tune_seed))
    fixed_diff, fixed_se = _paired(reward(FixedCascade(env.cheap.decision_threshold)), r_tuned)
    correct_diff, correct_se = _paired(r_correct, r_tuned)

    rows = []
    for axis, ratios in RATIOS.items():
        for r in ratios:
            agent = assumed(env, axis, r)
            if agent is None:
                continue
            r_mis = reward(PosteriorThreshold.from_params(agent), agent)
            regret, regret_se = _paired(r_correct, r_mis)
            diff, diff_se = _paired(r_mis, r_tuned)
            rows.append({"true_p": env.p_target, "axis": axis, "ratio": r, "regret": regret, "regret_se": regret_se,
                         "diff_vs_tuned": diff, "diff_vs_tuned_se": diff_se,
                         "correct_diff_vs_tuned": correct_diff, "correct_diff_vs_tuned_se": correct_se,
                         "fixed_diff_vs_tuned": fixed_diff, "fixed_diff_vs_tuned_se": fixed_se})
    return pd.DataFrame(rows)


def break_even(d: pd.DataFrame) -> dict:
    """Ratios on each side of r = 1 where the misspecified adaptive policy falls below the tuned cascade.

    Linear interpolation in log(r) between the last grid point with diff >= 0 and the
    first with diff < 0, walking outward from r = 1. NaN = never crosses within the grid.
    If the correctly specified advantage is not significant (95% CI includes 0), the
    break-even is not meaningful and is flagged.
    """
    d = d.sort_values("ratio")
    at1 = d.iloc[np.argmin(np.abs(np.log(d.ratio)))]
    significant = at1.correct_diff_vs_tuned - Z95 * at1.correct_diff_vs_tuned_se > 0

    def walk(side: pd.DataFrame) -> float:
        prev = None
        for _, row in side.iterrows():
            if row.diff_vs_tuned < 0:
                if prev is None:
                    return float(row.ratio)
                x0, x1 = np.log(prev.ratio), np.log(row.ratio)
                y0, y1 = prev.diff_vs_tuned, row.diff_vs_tuned
                return float(np.exp(x0 + (0 - y0) * (x1 - x0) / (y1 - y0)))
            prev = row
        return float("nan")

    return {"advantage_at_correct": at1.correct_diff_vs_tuned, "advantage_se": at1.correct_diff_vs_tuned_se,
            "advantage_significant": bool(significant),
            "break_even_low": walk(d[d.ratio <= 1.0 + 1e-9].iloc[::-1]),
            "break_even_high": walk(d[d.ratio >= 1.0 - 1e-9])}


def run(n_items: int, seed: int, tune_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([robustness_rows(ProblemParams().with_updates(p_target=p), n_items, seed, tune_seed)
                    for p in TRUE_P], ignore_index=True)
    be = pd.DataFrame([{"true_p": p, "axis": axis, **break_even(d)}
                       for (p, axis), d in df.groupby(["true_p", "axis"], sort=False)])

    csv, be_csv = csv_path("robustness"), csv_path("robustness_breakeven")
    df.to_csv(csv, index=False)
    be.to_csv(be_csv, index=False)

    print(df[["true_p", "axis", "ratio", "regret", "regret_se", "diff_vs_tuned", "diff_vs_tuned_se"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(be.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote results/{csv.name}, results/{be_csv.name}")
    return df, be


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
