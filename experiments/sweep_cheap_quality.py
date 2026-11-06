"""Sweep cheap-model quality: the score separation d' between classes.

The fixed cascade's score threshold stays at its default, so its cheap-model
false-positive rate stays at ~10% while the true-positive rate rises with d'.

    python experiments/sweep_cheap_quality.py

Writes results/sweep_cheap_quality.csv.
"""

from dataclasses import replace

from common import base_parser, csv_path, report, score_thresholds

from adaptive_model_routing import ProblemParams, sweep

VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
X_COL = "cheap_separation"


def make_params(sep: float) -> ProblemParams:
    d = ProblemParams()
    return d.with_updates(cheap=replace(d.cheap, separation=sep))


def run(n_items: int, seed: int, tune_seed: int):
    print("adaptive policy as score thresholds: reject below / accept above")
    for v in VALUES:
        lo, hi = score_thresholds(make_params(v))
        print(f"  d'={v:<4} reject x < {lo:6.2f}   accept x > {hi:6.2f}")
    summary = sweep(X_COL, VALUES, make_params, n_items, seed, tune_seed)
    csv = csv_path("sweep_cheap_quality")
    summary.to_csv(csv, index=False)
    report(summary, X_COL, csv)
    return summary


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
