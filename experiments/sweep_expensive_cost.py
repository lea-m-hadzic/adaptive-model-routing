"""Sweep the expensive model's per-query cost (cheap model cost stays 1).

    python experiments/sweep_expensive_cost.py

Writes results/sweep_expensive_cost.csv.
"""

from dataclasses import replace

from common import base_parser, csv_path, report, score_thresholds

from adaptive_model_routing import ProblemParams, sweep

VALUES = [1, 2, 5, 10, 20, 50, 100]
X_COL = "expensive_cost"


def make_params(c: float) -> ProblemParams:
    d = ProblemParams()
    return d.with_updates(expensive=replace(d.expensive, cost=float(c)))


def run(n_items: int, seed: int, tune_seed: int):
    print("adaptive policy as score thresholds: reject below / accept above")
    for v in VALUES:
        lo, hi = score_thresholds(make_params(v))
        print(f"  expensive_cost={v:<4} reject x < {lo:6.2f}   accept x > {hi:6.2f}")
    summary = sweep(X_COL, VALUES, make_params, n_items, seed, tune_seed)
    csv = csv_path("sweep_expensive_cost")
    summary.to_csv(csv, index=False)
    report(summary, X_COL, csv)
    return summary


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
