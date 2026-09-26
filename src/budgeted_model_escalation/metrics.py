"""Summary metrics from per-item simulation results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    """Metrics for one policy's per-item results."""
    s, a = df["state"].to_numpy(bool), df["accepted"].to_numpy(bool)
    tp, fp = np.sum(s & a), np.sum(~s & a)
    fn, tn = np.sum(s & ~a), np.sum(~s & ~a)
    reward = df["reward"].to_numpy(float)
    return {
        "n_items": len(df),
        "n_targets": int(s.sum()),
        "recall": _ratio(tp, tp + fn),
        "precision": _ratio(tp, tp + fp),
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, tp + fn),
        "avg_cost": float(df["cost"].mean()),
        "frac_expensive": float(df["used_expensive"].mean()),
        "terminal_reward": float((reward + df["cost"].to_numpy(float)).mean()),  # before query costs
        "expected_reward": float(reward.mean()),  # net of query costs
        "reward_stderr": float(reward.std(ddof=1) / np.sqrt(len(reward))),
    }


def paired_difference(results: pd.DataFrame, policy: str, reference: str, col: str = "reward") -> tuple[float, float]:
    """Mean and standard error of (policy - reference), paired item by item.

    All policies run on the same items, so the paired SE is far smaller than the
    SE of either policy's mean on its own.
    """
    wide = results.pivot(index="item", columns="policy", values=col)
    d = (wide[policy] - wide[reference]).to_numpy(float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def summarize_by_policy(results: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    """One row of metrics per policy, preserving the order policies were run in.

    If `reference` names a policy, also report each policy's paired reward
    difference against it (`diff_vs_ref` +/- `diff_vs_ref_se`).
    """
    order = list(dict.fromkeys(results["policy"]))
    rows = []
    for p in order:
        row = {"policy": p, **summarize(results[results["policy"] == p])}
        if reference is not None:
            row["diff_vs_ref"], row["diff_vs_ref_se"] = paired_difference(results, p, reference)
        rows.append(row)
    return pd.DataFrame(rows)
