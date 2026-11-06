"""Compare all policies under default parameters.

    python experiments/run_default.py
    python experiments/run_default.py --p-target 0.05 --expensive-cost 5

The tuned cascade picks its threshold on --tune-seed and is evaluated on --seed.
Differences are paired (same items) and reported with 95% CIs.
Writes results/default_comparison.csv.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from common import base_parser, csv_path, report, score_thresholds

from adaptive_model_routing import ProblemParams, compare


def build_params(p_target: float, cheap_separation: float, cheap_threshold: float, expensive_cost: float) -> ProblemParams:
    d = ProblemParams()
    return replace(
        d,
        p_target=p_target,
        cheap=replace(d.cheap, separation=cheap_separation, decision_threshold=cheap_threshold),
        expensive=replace(d.expensive, cost=expensive_cost),
    )


def describe_adaptive_policy(params: ProblemParams) -> None:
    lo, hi = score_thresholds(params)
    c = params.cheap
    print(f"cheap model: d'={c.separation}, fixed-cascade threshold x >= {c.decision_threshold} "
          f"(TPR={c.tpr:.3f}, FPR={c.fpr:.3f})")
    print(f"adaptive policy: reject x < {lo:.2f}, escalate {lo:.2f} <= x <= {hi:.2f}, accept x > {hi:.2f}")


def run(params: ProblemParams, n_items: int, seed: int, tune_seed: int, name: str = "default_comparison") -> pd.DataFrame:
    describe_adaptive_policy(params)
    summary = compare(params, n_items, seed, tune_seed)
    csv = csv_path(name)
    summary.to_csv(csv, index=False)
    report(summary, None, csv)
    return summary


def main() -> None:
    d = ProblemParams()
    p = base_parser(__doc__)
    p.add_argument("--p-target", type=float, default=d.p_target)
    p.add_argument("--cheap-separation", type=float, default=d.cheap.separation, help="cheap model d'")
    p.add_argument("--cheap-threshold", type=float, default=d.cheap.decision_threshold,
                   help="score threshold used by cheap_only and fixed_cascade")
    p.add_argument("--expensive-cost", type=float, default=d.expensive.cost)
    p.add_argument("--out-name", default="default_comparison")
    a = p.parse_args()
    params = build_params(a.p_target, a.cheap_separation, a.cheap_threshold, a.expensive_cost)
    run(params, a.n_items, a.seed, a.tune_seed, a.out_name)


if __name__ == "__main__":
    main()
