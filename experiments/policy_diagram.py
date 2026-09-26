"""Figure 1: the adaptive policy's reject / escalate / accept regions.

Top row: posterior P(s=1 | x) after the cheap score x, with the belief thresholds.
Bottom row: class-conditional score densities. Columns: p_target = 0.01 and 0.1.
Computed directly from the model (no simulation).

    python experiments/policy_diagram.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import output_path, score_thresholds
from style import BLUE, BLUE_TINT, INK, MUTED, POLICY, TEXT_WIDTH, save, top_legend

from budgeted_model_escalation import PosteriorThreshold, ProblemParams
from budgeted_model_escalation import belief as bel

P_TARGETS = [0.01, 0.1]
X_RANGE = (-3.0, 6.0)
REGIONS = [("Reject", "#f2f2f2"), ("Escalate", BLUE_TINT), ("Accept", "#e0e0e0")]


def run() -> list:
    x = np.linspace(*X_RANGE, 1000)
    fig, axes = plt.subplots(2, len(P_TARGETS), figsize=(TEXT_WIDTH, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    fixed_style = dict(color=POLICY["fixed_cascade"]["color"], linestyle=POLICY["fixed_cascade"]["linestyle"],
                       linewidth=1.0)
    for j, p in enumerate(P_TARGETS):
        params = ProblemParams().with_updates(p_target=p)
        pol = PosteriorThreshold.from_params(params)
        x_lo, x_hi = score_thresholds(params)
        top, bottom = axes[0, j], axes[1, j]
        bounds = [(X_RANGE[0], x_lo), (x_lo, x_hi), (x_hi, X_RANGE[1])]

        for ax in (top, bottom):
            for (a, b), (_, color) in zip(bounds, REGIONS):
                ax.axvspan(a, b, color=color, linewidth=0, zorder=0)
            for xb in (x_lo, x_hi):
                ax.axvline(xb, color=INK, linewidth=0.6, zorder=1)
            ax.axvline(params.cheap.decision_threshold, zorder=2, **fixed_style)
            ax.grid(False)

        # Posterior and belief thresholds.
        top.plot(x, bel.update(p, x, params.cheap), color=BLUE, linewidth=1.6, zorder=3,
                 label="Posterior $P(s{=}1 \\mid x)$")
        top.set_yscale("log")
        top.set_ylim(1e-4, 3)
        top.set_xlim(*X_RANGE)
        for b, lab in [(pol.low, "$b_{\\mathrm{low}}$"), (pol.accept_threshold, "$b^*$"),
                       (pol.high, "$b_{\\mathrm{high}}$")]:
            top.axhline(b, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
            top.annotate(lab, (X_RANGE[0], b), xytext=(2, 1.5), textcoords="offset points", fontsize=7,
                         color=MUTED, va="bottom")
        for (a, b), (name, color) in zip(bounds, REGIONS):
            top.text((a + b) / 2, 1.6, name, ha="center", va="center", fontsize=8, color=INK, zorder=4,
                     bbox=dict(facecolor=color, edgecolor="none", pad=1.0))
        top.set_title(f"({'ab'[j]}) Prevalence $p = {p:g}$")
        if j == 0:
            top.set_ylabel("Posterior belief")

        # Class-conditional score densities.
        dens = lambda mu: np.exp(-0.5 * (x - mu) ** 2) / np.sqrt(2 * np.pi)
        bottom.plot(x, dens(0.0), color="#8c8c8c", linewidth=1.2, label="$p(x \\mid s{=}0)$")
        bottom.plot(x, dens(params.cheap.separation), color=INK, linewidth=1.2, label="$p(x \\mid s{=}1)$")
        bottom.set_yticks([])
        bottom.set_ylim(0, 0.46)
        bottom.set_xlabel("Cheap-model score $x$")
        # x_low labelled to the left of its boundary, x_high to the right, so they never collide.
        for xb, sub, dx, ha in [(x_lo, "low", -2, "right"), (x_hi, "high", 2, "left")]:
            bottom.annotate(f"$x_{{\\mathrm{{{sub}}}}} = {xb:.2f}$", (xb, 0.44), xytext=(dx, 0),
                            textcoords="offset points", fontsize=7, color=INK, va="top", ha=ha)
        if j == 0:
            bottom.set_ylabel("Density")

        print(f"p={p}: reject x < {x_lo:.2f}, escalate {x_lo:.2f}..{x_hi:.2f}, accept x > {x_hi:.2f} "
              f"(b_low={pol.low:.4f}, b*={pol.accept_threshold:.4f}, b_high={pol.high:.4f})")

    fig.align_ylabels(axes[:, 0])
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fixed = plt.Line2D([], [], **fixed_style)
    fig.tight_layout(h_pad=0.6, w_pad=1.5)
    top_legend(fig, h1 + [fixed] + h2, l1 + ["Fixed-cascade threshold $t$"] + l2, ncol=4)
    figs = save(fig, output_path("policy_diagram", "png"))
    print("wrote " + ", ".join(f.name for f in figs))
    return figs


if __name__ == "__main__":
    run()
