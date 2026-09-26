"""Shared helpers for experiment scripts: CLI, output paths, plotting, LaTeX tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from style import BLUE, INK, LABEL, MUTED, POLICY, TEXT_WIDTH, VERMILLION, line_kw, save, top_legend

from budgeted_model_escalation import PosteriorThreshold, ProblemParams

ROOT = Path(__file__).resolve().parents[1]
Z95 = 1.96


def base_parser(doc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-items", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0, help="evaluation seed")
    p.add_argument("--tune-seed", type=int, default=1, help="seed for tuning the cascade threshold")
    return p


def output_path(name: str, kind: str) -> Path:
    """kind: 'csv' or 'tex' -> results/, 'png' -> figures/."""
    folder = ROOT / ("figures" if kind == "png" else "results")
    folder.mkdir(exist_ok=True)
    return folder / f"{name}.{kind}"


def score_thresholds(params: ProblemParams) -> tuple[float, float]:
    """The adaptive policy's belief thresholds, expressed as cheap-score thresholds.

    Returns (reject_below, accept_above): scores in between are escalated.
    """
    pol = PosteriorThreshold.from_params(params)
    to_score = lambda b: params.cheap.score_at_posterior(params.p_target, b)
    return to_score(pol.low), to_score(pol.high)


def plain_log_ticks(ax, ticks, axis: str = "x") -> None:
    """Log-scaled axis with plain-number tick labels at `ticks` (no scientific notation)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    a.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
    a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def reference_line(ax, y: float = 0.0, label: str | None = None) -> None:
    """The tuned cascade as a horizontal reference at `y` (differences are measured against it)."""
    ax.axhline(y, color=VERMILLION, linewidth=1.2, zorder=2, label=label)


def ci_band(ax, x, y, se, color) -> None:
    ax.fill_between(x, y - Z95 * se, y + Z95 * se, color=color, alpha=0.16, linewidth=0, zorder=1)


def ci_handle() -> Patch:
    return Patch(facecolor=BLUE, alpha=0.16, label="95% CI (paired)")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def num(v: float, fmt: str = "{:.2f}", bold: bool = False) -> str:
    """A number in math mode, so negative signs typeset as minus signs."""
    body = fmt.format(v)
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def fmt_pm(mean: float, se: float, digits: int = 2, bold: bool = False) -> str:
    """Mean with a 95% CI half-width, e.g. $+0.04 \\pm 0.06$; bold if requested."""
    m, h = f"{mean:+.{digits}f}", f"{Z95 * se:.{digits}f}"
    return f"$\\mathbf{{{m}}} \\pm \\mathbf{{{h}}}$" if bold else f"${m} \\pm {h}$"


def significant(mean: float, se: float) -> bool:
    """95% CI excludes zero."""
    return abs(mean) > Z95 * se


def write_latex_table(df: pd.DataFrame, path: Path, columns: dict[str, str], align: str | None = None) -> None:
    """Minimal booktabs tabular (no jinja2 dependency). Cells must already be formatted strings.

    `columns` maps df column -> header.
    """
    align = align or "l" + "r" * (len(columns) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(columns.values()) + r" \\", r"\midrule"]
    prev_group = None
    for _, row in df.iterrows():
        group = row.get("_group")
        if prev_group is not None and group != prev_group:
            lines.append(r"\addlinespace")
        prev_group = group
        lines.append(" & ".join(str(row[c]) for c in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Sweep figures
# ---------------------------------------------------------------------------

SWEEP_PANELS = [
    ("expected_reward", "Net reward per item"),
    ("recall", "Recall"),
    ("avg_cost", "Query cost per item"),
    ("frac_expensive", "Fraction escalated"),
]
PAPER_POLICIES = ["cheap_only", "expensive_only", "fixed_cascade", "tuned_cascade", "posterior_threshold"]


def plot_sweep(summary: pd.DataFrame, x_col: str, x_label: str, path: Path, log_x=None) -> list[Path]:
    """Supplementary figure: 2x2 panels (one per metric), one line per policy."""
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.2), sharex=True)
    for ax, (col, title) in zip(axes.flat, SWEEP_PANELS):
        for policy in PAPER_POLICIES:
            d = summary[summary["policy"] == policy]
            ax.plot(d[x_col], d[col], **line_kw(policy))
        if log_x:
            plain_log_ticks(ax, log_x)
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel(x_label)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    top_legend(fig, h[::-1], l[::-1], ncol=5)
    return save(fig, path)


def _top_row_limits(ada_lo: np.ndarray, ada_hi: np.ndarray, others: np.ndarray) -> tuple[float, float]:
    """y-limits that keep the adaptive CI and nearby baseline values in view.

    Baseline values more than 3x the adaptive range below zero are drawn at the
    bottom edge with their value printed, instead of squashing the adaptive curve.
    """
    top = max(float(ada_hi.max()), 0.0)
    span = max(top, float(-ada_lo.min()), 0.25)
    bottom = min(float(ada_lo.min()), float(others[others >= -3 * span].min(initial=0.0)))
    pad = 0.08 * (top - bottom)
    return bottom - pad, top + pad


def plot_sweep_summary(sweeps: list[tuple[pd.DataFrame, str, str, bool, str]], path: Path) -> list[Path]:
    """Paper figure: columns = sweeps; top = net reward vs tuned cascade, bottom = fraction escalated.

    `sweeps` holds (summary, x_col, x_label, log_ticks, column_title) per column; log_ticks is a
    list of tick positions for a log axis, or None for a linear axis.
    """
    n = len(sweeps)
    fig, axes = plt.subplots(2, n, figsize=(TEXT_WIDTH, 3.9), sharex="col",
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for j, (summary, x_col, x_label, log_x, col_title) in enumerate(sweeps):
        top, bottom = axes[0, j], axes[1, j]
        get = lambda p: summary[summary["policy"] == p].sort_values(x_col)

        ada, fixed = get("posterior_threshold"), get("fixed_cascade")
        x = ada[x_col].to_numpy()
        y, se = ada["diff_vs_ref"].to_numpy(), ada["diff_vs_ref_se"].to_numpy()
        ylo, yhi = _top_row_limits(y - Z95 * se, y + Z95 * se, fixed["diff_vs_ref"].to_numpy())

        reference_line(top, label=LABEL["tuned_cascade"] + " (reference)")
        fx, fy = fixed[x_col].to_numpy(), fixed["diff_vs_ref"].to_numpy()
        clipped = fy < ylo
        top.plot(fx, np.maximum(fy, ylo), **line_kw("fixed_cascade", markers=False))
        top.plot(fx[~clipped], fy[~clipped], **line_kw("fixed_cascade", linestyle="none", label="_"))
        for xc, yc in zip(fx[clipped], fy[clipped]):
            # Off-scale value: arrow marker on the bottom edge, value printed beside it.
            top.plot(xc, ylo, marker="v", color=POLICY["fixed_cascade"]["color"], markersize=4, clip_on=False,
                     zorder=6, linestyle="none")
            top.annotate(f"${yc:.1f}$", (xc, ylo), xytext=(-7, 5), textcoords="offset points", ha="right",
                         va="bottom", fontsize=7, color=MUTED, zorder=7,
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ci_band(top, x, y, se, BLUE)
        top.plot(x, y, **line_kw("posterior_threshold"))
        top.set_ylim(ylo, yhi)
        top.set_title(col_title)

        for p in ["fixed_cascade", "tuned_cascade", "posterior_threshold"]:
            d = get(p)
            bottom.plot(d[x_col], d["frac_expensive"], **line_kw(p, markers=(p != "fixed_cascade")))
        bottom.set_ylim(0, None)
        bottom.set_xlabel(x_label)
        for ax in (top, bottom):
            if log_x:
                plain_log_ticks(ax, log_x)

    axes[0, 0].set_ylabel("$\\Delta$ net reward vs.\ntuned cascade")
    axes[1, 0].set_ylabel("Fraction escalated")
    fig.align_ylabels(axes[:, 0])
    handles = [plt.Line2D([], [], **line_kw("posterior_threshold")), ci_handle(),
               plt.Line2D([], [], **line_kw("tuned_cascade")),
               plt.Line2D([], [], **line_kw("fixed_cascade", markers=False))]
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    top_legend(fig, handles, [h.get_label() for h in handles], ncol=4)
    return save(fig, path)


def report(summary: pd.DataFrame, x_col: str | None, *paths: Path) -> None:
    cols = ([x_col] if x_col else []) + ["policy", "recall", "precision", "avg_cost", "frac_expensive",
                                          "expected_reward", "diff_vs_ref", "diff_vs_ref_se"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    flat = [q for p in paths for q in (p if isinstance(p, list) else [p])]
    print("\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in flat))
