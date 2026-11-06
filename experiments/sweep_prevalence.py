"""Sweep target prevalence p_target (all other parameters at defaults).

    python experiments/sweep_prevalence.py

Writes results/sweep_prevalence.csv.
"""

from common import base_parser, csv_path, report, score_thresholds

from adaptive_model_routing import ProblemParams, sweep

VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
X_COL = "p_target"


def make_params(p: float) -> ProblemParams:
    return ProblemParams().with_updates(p_target=p)


def run(n_items: int, seed: int, tune_seed: int):
    print("adaptive policy as score thresholds: reject below / accept above")
    for v in VALUES:
        lo, hi = score_thresholds(make_params(v))
        print(f"  p_target={v:<6} reject x < {lo:6.2f}   accept x > {hi:6.2f}")
    summary = sweep(X_COL, VALUES, make_params, n_items, seed, tune_seed)
    csv = csv_path("sweep_prevalence")
    summary.to_csv(csv, index=False)
    report(summary, X_COL, csv)
    return summary


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
