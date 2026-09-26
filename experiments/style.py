"""Shared figure theme: fonts, sizes, policy colors/line styles, and saving.

Design rules:
- Computer Modern text and math, to match a LaTeX paper; fonts embedded as TrueType.
- Figures are drawn at their final printed size (full text width = 6.5 in), so font
  sizes in points are the sizes the reader sees.
- Only the two methods being compared carry hue: adaptive (blue) and tuned cascade
  (vermillion), from the Okabe-Ito colorblind-safe set. All other baselines are
  grays, told apart by lightness, marker, and dash pattern.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_WIDTH = 6.5  # inches

INK = "#1a1a1a"
MUTED = "#555555"
GRID = "#e6e6e6"
BLUE = "#0072B2"       # Okabe-Ito blue
VERMILLION = "#D55E00"  # Okabe-Ito vermillion
BLUE_TINT = "#dbe9f4"
VERMILLION_TINT = "#f6e0d0"

# Paper-facing names.
LABEL = {
    "posterior_threshold": "Adaptive (VOI)",
    "tuned_cascade": "Tuned cascade",
    "fixed_cascade": "Fixed cascade",
    "expensive_only": "Expensive only",
    "cheap_only": "Cheap only",
}

# One visual identity per policy, used in every figure.
POLICY = {
    "posterior_threshold": dict(color=BLUE, linestyle="-", marker="o", linewidth=1.8, markersize=4, zorder=5),
    "tuned_cascade": dict(color=VERMILLION, linestyle="-", marker="s", linewidth=1.4, markersize=3.8, zorder=4),
    "fixed_cascade": dict(color="#7a7a7a", linestyle=(0, (4, 2)), marker="v", linewidth=1.1, markersize=4, zorder=3),
    "expensive_only": dict(color="#3f3f3f", linestyle=(0, (1, 1.5)), marker="^", linewidth=1.1, markersize=4, zorder=2),
    "cheap_only": dict(color="#a6a6a6", linestyle=(0, (1, 1.5)), marker="D", linewidth=1.1, markersize=3.4, zorder=2),
}


def line_kw(policy: str, markers: bool = True, **overrides) -> dict:
    """Keyword arguments for ax.plot for one policy's line."""
    kw = dict(POLICY[policy], label=LABEL[policy])
    if markers:
        kw.update(markeredgecolor="white", markeredgewidth=0.6)
    else:
        kw.pop("marker")
        kw.pop("markersize")
    kw.update(overrides)
    return kw


def point_kw(policy: str, size: float = 6.5) -> dict:
    """Keyword arguments for a single reference point (no line)."""
    s = POLICY[policy]
    return dict(color=s["color"], marker=s["marker"], markersize=size, linestyle="none",
                markeredgecolor="white", markeredgewidth=0.8, zorder=6, label=LABEL[policy])


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,  # cmr10 has no unicode minus; use math minus
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titlelocation": "left",
        "axes.titlepad": 5,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.columnspacing": 1.4,
        "lines.solid_capstyle": "round",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,  # TrueType, not Type 3 (conference PDF checkers reject Type 3)
        "ps.fonttype": 42,
    })


def save(fig, stem: Path) -> list[Path]:
    """Save `stem`.png (preview) and `stem`.pdf (paper), then close the figure."""
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def top_legend(fig, handles, labels, ncol: int, y: float = 1.0) -> None:
    """One shared, frameless legend centered above all panels."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               handletextpad=0.5, borderaxespad=0.0)


apply()
