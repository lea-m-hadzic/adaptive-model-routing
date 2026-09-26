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

Writes results/robustness.csv, results/robustness_breakeven.{csv,tex}, figures/robustness.png.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (Z95, base_parser, ci_band, ci_handle, fmt_pm, num, output_path, plain_log_ticks,
                    reference_line, write_latex_table)
from style import BLUE, BLUE_TINT, INK, LABEL, MUTED, TEXT_WIDTH, line_kw, save, top_legend

from budgeted_model_escalation import FixedCascade, PosteriorThreshold, ProblemParams
from budgeted_model_escalation.environment import sample_items
from budgeted_model_escalation.simulation import run_policy, tune_cascade

TRUE_P = [0.01, 0.1]
RATIOS = {
    "prevalence": np.geomspace(0.1, 10, 11),
    "cheap_dprime": np.geomspace(0.5, 2, 11),
}
TICKS = {"prevalence": [0.1, 0.3, 1, 3, 10], "cheap_dprime": [0.5, 0.7, 1, 1.4, 2]}
AXIS_LABEL = {"prevalence": "Assumed / true prevalence, $r$", "cheap_dprime": "Assumed / true cheap-model $d'$, $r$"}


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


def tolerated_range(b: pd.Series) -> str:
    """'$[0.38, 2.82]$'; a bound that never crosses within the grid shows as the grid edge (<= or >=)."""
    grid = RATIOS[b.axis]
    lo = f"{b.break_even_low:.2f}" if np.isfinite(b.break_even_low) else f"\\le {grid.min():.2f}"
    hi = f"{b.break_even_high:.2f}" if np.isfinite(b.break_even_high) else f"\\ge {grid.max():.2f}"
    return f"$[{lo}, {hi}]$" + ("" if b.advantage_significant else r"$^\dagger$")


def plot(df: pd.DataFrame, be: pd.DataFrame, stem) -> list:
    axes_names = list(RATIOS)
    fig, axes = plt.subplots(len(TRUE_P), len(axes_names), figsize=(TEXT_WIDTH, 4.5), sharex="col", sharey="row")
    letters = iter("abcdefgh")
    for i, p in enumerate(TRUE_P):
        for j, axis in enumerate(axes_names):
            ax = axes[i, j]
            d = df[(df.true_p == p) & (df.axis == axis)].sort_values("ratio")
            b = be[(be.true_p == p) & (be.axis == axis)].iloc[0]
            x, y, se = d.ratio.to_numpy(), d.diff_vs_tuned.to_numpy(), d.diff_vs_tuned_se.to_numpy()

            if b.advantage_significant:
                lo = b.break_even_low if np.isfinite(b.break_even_low) else x.min()
                hi = b.break_even_high if np.isfinite(b.break_even_high) else x.max()
                ax.axvspan(lo, hi, color=BLUE_TINT, alpha=0.7, linewidth=0, zorder=0,
                           label="Tolerated range (above tuned cascade)")
                for r in (b.break_even_low, b.break_even_high):
                    if np.isfinite(r):
                        ax.plot(r, 0, marker="o", markersize=4, markerfacecolor="white", markeredgecolor=INK,
                                markeredgewidth=0.8, zorder=7, linestyle="none")
                        ax.annotate(f"${r:.2f}$", (r, 0), xytext=(0, 5), textcoords="offset points", ha="center",
                                    fontsize=7, color=INK)
            else:
                ax.text(0.03, 0.06, "No significant advantage at $r = 1$", transform=ax.transAxes, fontsize=7,
                        color=MUTED)
            ax.axvline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
            reference_line(ax, label=LABEL["tuned_cascade"] + " (reference)")
            ax.axhline(d.fixed_diff_vs_tuned.iloc[0], **line_kw("fixed_cascade", markers=False))
            ci_band(ax, x, y, se, BLUE)
            ax.plot(x, y, **line_kw("posterior_threshold", markersize=3.2,
                                    label="Adaptive with assumed ratio $r$"))
            plain_log_ticks(ax, TICKS[axis])
            what = "prevalence" if axis == "prevalence" else "cheap-model $d'$"
            ax.set_title(f"({next(letters)}) Misspecified {what}, true $p = {p:g}$")
            if i == len(TRUE_P) - 1:
                ax.set_xlabel(AXIS_LABEL[axis])
        axes[i, 0].set_ylabel("$\\Delta$ net reward vs.\ntuned cascade")
    fig.align_ylabels(axes[:, 0])
    handles, labels = axes[1, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ["Adaptive with assumed ratio $r$", "Tolerated range (above tuned cascade)",
             LABEL["tuned_cascade"] + " (reference)", LABEL["fixed_cascade"]]
    handles = [by_label[k] for k in order]
    handles.insert(1, ci_handle())
    order.insert(1, ci_handle().get_label())
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    top_legend(fig, handles, order, ncol=3)
    return save(fig, stem)


def run(n_items: int, seed: int, tune_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([robustness_rows(ProblemParams().with_updates(p_target=p), n_items, seed, tune_seed)
                    for p in TRUE_P], ignore_index=True)
    be = pd.DataFrame([{"true_p": p, "axis": axis, **break_even(d)}
                       for (p, axis), d in df.groupby(["true_p", "axis"], sort=False)])

    csv, be_csv, be_tex, png = (output_path("robustness", "csv"), output_path("robustness_breakeven", "csv"),
                                output_path("robustness_breakeven", "tex"), output_path("robustness", "png"))
    df.to_csv(csv, index=False)
    be.to_csv(be_csv, index=False)
    # Table 3: tolerated misspecification range; dagger = no significant advantage at r = 1,
    # so the break-even points are within noise.
    t = pd.DataFrame({
        "true_p": [num(v, "{:g}") for v in be.true_p],
        "axis": be.axis.map({"prevalence": "Prevalence", "cheap_dprime": "Cheap-model $d'$"}),
        "adv": [fmt_pm(m, s, bold=sig and m > 0) for m, s, sig in
                zip(be.advantage_at_correct, be.advantage_se, be.advantage_significant)],
        "range": [tolerated_range(r) for _, r in be.iterrows()],
    })
    write_latex_table(t, be_tex, columns={"true_p": "True $p$", "axis": "Misspecified parameter",
                                          "adv": r"Advantage at $r=1$ (95\% CI)",
                                          "range": r"Tolerated range of $r$"}, align="llrr")
    with open(be_tex, "a") as f:
        f.write("% dagger: advantage at r = 1 is not significant, so the break-even ratios are within noise.\n")
    figs = plot(df, be, png)

    print(df[["true_p", "axis", "ratio", "regret", "regret_se", "diff_vs_tuned", "diff_vs_tuned_se"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(be.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {csv.name}, {be_csv.name}, {be_tex.name}, " + ", ".join(f.name for f in figs))
    return df, be


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
