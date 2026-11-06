"""Shared helpers for experiment scripts: CLI, output paths, and printing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adaptive_model_routing import PosteriorThreshold, ProblemParams

ROOT = Path(__file__).resolve().parents[1]
Z95 = 1.96


def base_parser(doc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-items", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0, help="evaluation seed")
    p.add_argument("--tune-seed", type=int, default=1, help="seed for tuning the cascade threshold")
    return p


def csv_path(name: str) -> Path:
    """results/<name>.csv, creating results/ if needed."""
    folder = ROOT / "results"
    folder.mkdir(exist_ok=True)
    return folder / f"{name}.csv"


def score_thresholds(params: ProblemParams) -> tuple[float, float]:
    """The adaptive policy's belief thresholds, expressed as cheap-score thresholds.

    Returns (reject_below, accept_above): scores in between are escalated.
    """
    pol = PosteriorThreshold.from_params(params)
    to_score = lambda b: params.cheap.score_at_posterior(params.p_target, b)
    return to_score(pol.low), to_score(pol.high)


def report(summary: pd.DataFrame, x_col: str | None, path: Path) -> None:
    cols = ([x_col] if x_col else []) + ["policy", "recall", "precision", "avg_cost", "frac_expensive",
                                          "expected_reward", "diff_vs_ref", "diff_vs_ref_se"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {path.relative_to(ROOT)}")
