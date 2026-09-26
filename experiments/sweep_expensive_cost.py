"""Sweep the expensive model's per-query cost (cheap model cost stays 1).

    python experiments/sweep_expensive_cost.py
"""

from dataclasses import replace

from common import base_parser, output_path, plot_sweep, report, score_thresholds

from budgeted_model_escalation import ProblemParams, sweep

VALUES = [1, 2, 5, 10, 20, 50, 100]
X_COL, X_LABEL, LOG_X = "expensive_cost", "Expensive-model cost $c_{\\mathrm{exp}}$", [1, 10, 100]
TABLE_LABEL = r"Cost $c_{\mathrm{exp}}$"
TITLE = "Expensive-model cost"


def make_params(c: float) -> ProblemParams:
    d = ProblemParams()
    return d.with_updates(expensive=replace(d.expensive, cost=float(c)))


def run(n_items: int, seed: int, tune_seed: int):
    print("adaptive policy as score thresholds: reject below / accept above")
    for v in VALUES:
        lo, hi = score_thresholds(make_params(v))
        print(f"  expensive_cost={v:<4} reject x < {lo:6.2f}   accept x > {hi:6.2f}")
    summary = sweep(X_COL, VALUES, make_params, n_items, seed, tune_seed)
    csv, png = output_path("sweep_expensive_cost", "csv"), output_path("sweep_expensive_cost", "png")
    summary.to_csv(csv, index=False)
    figs = plot_sweep(summary, X_COL, X_LABEL, png, log_x=LOG_X)
    report(summary, X_COL, csv, figs)
    return summary


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
