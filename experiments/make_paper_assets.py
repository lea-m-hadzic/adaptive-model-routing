"""Regenerate every figure and table used in the paper, from scratch.

    python experiments/make_paper_assets.py            # full run (~2 min)
    python experiments/make_paper_assets.py --quick    # 20k items, for a smoke test

Each figure is written as .png (preview, 300 dpi) and .pdf (for the paper).

Paper assets:
    Fig 1  figures/policy_diagram.png      reject / escalate / accept regions
    Tab 1  results/default_comparison.tex  default setting, paired diffs vs tuned cascade
    Fig 2  figures/frontier.png            terminal reward & recall vs query cost
    Fig 3  figures/sweeps_summary.png      prevalence / cost / cheap-quality sweeps
    Tab 2  results/sweeps_summary.tex      numbers behind Fig 3
    Fig 4  figures/robustness.png          misspecified prevalence and cheap-model d'
    Tab 3  results/robustness_breakeven.tex break-even misspecification ratios
Supplementary: figures/default_comparison.png, figures/sweep_*.png, all CSVs.
"""

from __future__ import annotations

import time

import pandas as pd
from common import base_parser, fmt_pm, num, output_path, plot_sweep_summary, significant, write_latex_table

import frontier
import policy_diagram
import robustness
import run_default
import sweep_cheap_quality
import sweep_expensive_cost
import sweep_prevalence
from adaptive_model_routing import ProblemParams

SWEEPS = [(sweep_prevalence, "Prevalence"), (sweep_expensive_cost, "Expensive cost"),
          (sweep_cheap_quality, "Cheap $d'$")]


def sweeps_summary_table(summaries: list[pd.DataFrame], path) -> None:
    """Table 2: adaptive vs tuned cascade across sweeps; bold = adaptive significantly better."""
    rows = []
    for (mod, _), s in zip(SWEEPS, summaries):
        for v, d in s.groupby(mod.X_COL, sort=False):
            d = d.set_index("policy")
            ada, tuned = d.loc["posterior_threshold"], d.loc["tuned_cascade"]
            better = ada.diff_vs_ref > 0 and significant(ada.diff_vs_ref, ada.diff_vs_ref_se)
            rows.append({"_group": mod.TITLE, "sweep": mod.TABLE_LABEL, "value": num(v, "{:g}"),
                         "diff": fmt_pm(ada.diff_vs_ref, ada.diff_vs_ref_se, bold=better),
                         "esc_ada": num(ada.frac_expensive), "esc_tuned": num(tuned.frac_expensive),
                         "cost_ada": num(ada.avg_cost), "cost_tuned": num(tuned.avg_cost)})
    t = pd.DataFrame(rows)
    t.loc[t["_group"].duplicated(), "sweep"] = ""  # print each sweep name once
    write_latex_table(
        t, path, align="llrrrrr",
        columns={"sweep": "Sweep", "value": "Value", "diff": r"$\Delta$ net reward (95\% CI)",
                 "esc_ada": "Esc.\\ (adapt.)", "esc_tuned": "Esc.\\ (tuned)",
                 "cost_ada": "Cost (adapt.)", "cost_tuned": "Cost (tuned)"},
    )


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--quick", action="store_true", help="use 20k items per run")
    a = p.parse_args()
    n = 20_000 if a.quick else a.n_items
    start = time.time()

    def section(title: str) -> None:
        print(f"\n{'=' * 70}\n{title}   [{time.time() - start:.0f}s]\n{'=' * 70}")

    section("Fig 1: policy diagram")
    policy_diagram.run()

    section("Table 1: default comparison")
    run_default.run(ProblemParams(), n, a.seed, a.tune_seed)

    section("Fig 2: budget frontier")
    frontier.run(n, a.seed, a.tune_seed)

    summaries = []
    for mod, name in SWEEPS:
        section(f"Sweep: {name}")
        summaries.append(mod.run(n, a.seed, a.tune_seed))
    section("Fig 3 / Table 2: sweep summary")
    figs = plot_sweep_summary([(s, mod.X_COL, mod.X_LABEL, mod.LOG_X, f"({'abc'[k]}) {mod.TITLE}")
                               for k, ((mod, _), s) in enumerate(zip(SWEEPS, summaries))],
                              output_path("sweeps_summary", "png"))
    print("wrote " + ", ".join(f.name for f in figs))
    sweeps_summary_table(summaries, output_path("sweeps_summary", "tex"))

    section("Fig 4 / Table 3: robustness")
    robustness.run(n, a.seed, a.tune_seed)

    section(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
