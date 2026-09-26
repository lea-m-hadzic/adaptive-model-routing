"""Experiment 1: compare all policies under default parameters (Table 1).

    python experiments/run_default.py
    python experiments/run_default.py --p-target 0.05 --expensive-cost 5

The tuned cascade picks its threshold on --tune-seed and is evaluated on --seed.
Differences are paired (same items) and reported with 95% CIs.
Writes results/default_comparison.{csv,tex} and figures/default_comparison.png.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import pandas as pd
from common import (Z95, base_parser, fmt_pm, num, output_path, report, score_thresholds, significant,
                    write_latex_table)
from style import INK, LABEL, MUTED, POLICY, TEXT_WIDTH, save

from budgeted_model_escalation import ProblemParams, compare


def build_params(p_target: float, cheap_separation: float, cheap_threshold: float, expensive_cost: float) -> ProblemParams:
    d = ProblemParams()
    return replace(
        d,
        p_target=p_target,
        cheap=replace(d.cheap, separation=cheap_separation, decision_threshold=cheap_threshold),
        expensive=replace(d.expensive, cost=expensive_cost),
    )


def describe_adaptive_policy(params: ProblemParams) -> None:
    lo, hi = score_thresholds(params)
    c = params.cheap
    print(f"cheap model: d'={c.separation}, fixed-cascade threshold x >= {c.decision_threshold} "
          f"(TPR={c.tpr:.3f}, FPR={c.fpr:.3f})")
    print(f"adaptive policy: reject x < {lo:.2f}, escalate {lo:.2f} <= x <= {hi:.2f}, accept x > {hi:.2f}")


def plot(summary: pd.DataFrame, stem) -> list:
    """Supplementary figure: one horizontal bar per policy for four metrics."""
    panels = [
        ("diff_vs_ref", "$\\Delta$ net reward vs. tuned cascade"),
        ("recall", "Recall"),
        ("avg_cost", "Query cost per item"),
        ("frac_expensive", "Fraction escalated"),
    ]
    names = summary["policy"].tolist()
    colors = [POLICY[n]["color"] for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.6), sharey=True)
    for ax, (col, title) in zip(axes.flat, panels):
        vals = summary[col].to_numpy()
        err = Z95 * summary["diff_vs_ref_se"].to_numpy() if col == "diff_vs_ref" else None
        ax.barh(range(len(names)), vals, xerr=err, color=colors, height=0.62,
                error_kw=dict(ecolor=INK, lw=0.8, capsize=2, capthick=0.8))
        ax.axvline(0, color=MUTED, linewidth=0.6)
        for y, v in enumerate(vals):
            ax.annotate(f"${v:.2f}$", (v, y), xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=7, color=MUTED)
        ax.set_yticks(range(len(names)), [LABEL[n] for n in names])
        ax.invert_yaxis()
        ax.margins(x=0.22)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.tight_layout(h_pad=1.4, w_pad=1.5)
    return save(fig, stem)


def write_table(summary: pd.DataFrame, path) -> None:
    """Table 1. Delta is paired vs the tuned cascade with a 95% CI; bold = significantly better."""
    t = pd.DataFrame({
        "policy": summary["policy"].map(LABEL),
        "recall": [num(v, "{:.3f}") for v in summary["recall"]],
        "precision": [num(v, "{:.3f}") for v in summary["precision"]],
        "cost": [num(v) for v in summary["avg_cost"]],
        "esc": [num(v) for v in summary["frac_expensive"]],
        "net": [num(v) for v in summary["expected_reward"]],
        "diff": ["---" if p == "tuned_cascade" else fmt_pm(m, s, bold=m > 0 and significant(m, s))
                 for p, m, s in zip(summary["policy"], summary["diff_vs_ref"], summary["diff_vs_ref_se"])],
    })
    write_latex_table(t, path, columns={
        "policy": "Policy", "recall": "Recall", "precision": "Precision", "cost": "Cost / item",
        "esc": "Escalated", "net": "Net reward", "diff": r"$\Delta$ vs.\ tuned (95\% CI)"})


def run(params: ProblemParams, n_items: int, seed: int, tune_seed: int, name: str = "default_comparison") -> pd.DataFrame:
    describe_adaptive_policy(params)
    summary = compare(params, n_items, seed, tune_seed)
    csv, tex, png = output_path(name, "csv"), output_path(name, "tex"), output_path(name, "png")
    summary.to_csv(csv, index=False)
    write_table(summary, tex)
    figs = plot(summary, png)
    report(summary, None, csv, tex, figs)
    return summary


def main() -> None:
    d = ProblemParams()
    p = base_parser(__doc__)
    p.add_argument("--p-target", type=float, default=d.p_target)
    p.add_argument("--cheap-separation", type=float, default=d.cheap.separation, help="cheap model d'")
    p.add_argument("--cheap-threshold", type=float, default=d.cheap.decision_threshold,
                   help="score threshold used by cheap_only and fixed_cascade")
    p.add_argument("--expensive-cost", type=float, default=d.expensive.cost)
    p.add_argument("--out-name", default="default_comparison")
    a = p.parse_args()
    params = build_params(a.p_target, a.cheap_separation, a.cheap_threshold, a.expensive_cost)
    run(params, a.n_items, a.seed, a.tune_seed, a.out_name)


if __name__ == "__main__":
    main()
