"""Sweep target prevalence p_target (all other parameters at defaults).

    python experiments/sweep_prevalence.py
"""

from common import base_parser, output_path, plot_sweep, report, score_thresholds

from adaptive_model_routing import ProblemParams, sweep

VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
X_COL, X_LABEL, LOG_X = "p_target", "Target prevalence $p$", [0.001, 0.01, 0.1]
TABLE_LABEL = r"Prevalence $p$"
TITLE = "Prevalence"


def make_params(p: float) -> ProblemParams:
    return ProblemParams().with_updates(p_target=p)


def run(n_items: int, seed: int, tune_seed: int):
    print("adaptive policy as score thresholds: reject below / accept above")
    for v in VALUES:
        lo, hi = score_thresholds(make_params(v))
        print(f"  p_target={v:<6} reject x < {lo:6.2f}   accept x > {hi:6.2f}")
    summary = sweep(X_COL, VALUES, make_params, n_items, seed, tune_seed)
    csv, png = output_path("sweep_prevalence", "csv"), output_path("sweep_prevalence", "png")
    summary.to_csv(csv, index=False)
    figs = plot_sweep(summary, X_COL, X_LABEL, png, log_x=LOG_X)
    report(summary, X_COL, csv, figs)
    return summary


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    run(a.n_items, a.seed, a.tune_seed)
