<div align="center">

#  Budgeted Model Escalation for Rare-Target Search

</div>

Adaptive model escalation for rare-target detection under limited inference budgets.
CS 238 / AA 228 final project. Fully synthetic: no datasets, APIs, or trained models.

**Question:** on a rare-target detection task, when should an agent trust a cheap, noisy
classifier, and when should it pay to query an expensive, more accurate one?

**Central claim.** Model-based escalation matches a well-tuned cascade without requiring
threshold tuning, and can strictly improve on it when sufficiently confident cheap-model
predictions can be accepted without escalation. Its advantage depends on the accuracy of the
assumed observation model.

## Layout

```
src/adaptive_model_routing/
  models.py       
  belief.py   
  environment.py  
  policies.py  
  simulation.py   
  metrics.py     
experiments/
  make_paper_assets.py   
  policy_diagram.py
  run_default.py
  frontier.py
  robustness.py
  sweep_*.py
  style.py               
  common.py              
tests/           
```

## Model

Each item is a small POMDP that ends when the agent accepts or rejects.

| | |
|---|---|
| State | $s \in \{0,1\}$, hidden; $P(s=1) = p_{\text{target}}$ (small) |
| Actions | `query_cheap`, `query_expensive`, `accept_target`, `reject_target` |
| Cheap observation | real-valued score $x$ with $x\mid s{=}0 \sim \mathcal N(0,1)$ and $x\mid s{=}1 \sim \mathcal N(d',1)$ |
| Expensive observation | $o \in \{+,-\}$ with $P(o=+\mid s=1)=\text{TPR}$ and $P(o=+\mid s=0)=\text{FPR}$ |
| Belief | $b = P(s=1 \mid \text{history})$, starting at $b_0 = p_{\text{target}}$ |

The cheap model's quality is a single number, the separation $d'$ between the class means
(ROC AUC $= \Phi(d'/\sqrt2)$). Think of $x$ as the model's logit.

**Belief update** (Bayes' rule, with a likelihood $p(o\mid s)$ that is a probability for the
binary model and a density for the score):

$$b' = \frac{p(o\mid s=1)\,b}{p(o\mid s=1)\,b + p(o\mid s=0)\,(1-b)}$$

For the Gaussian score this is linear in log-odds:

$$\text{logit}(b') = \text{logit}(b) + d'x - d'^2/2.$$

Because the posterior is monotone in $x$, every belief threshold maps to a score threshold.

**Reward.** A query costs $c_m$. The final decision earns $R_{TP}$ for a true positive,
$-C_{FP}$ for a false positive, $-C_{FN}$ for a false negative, and $0$ for a true negative.

Assumptions:
- Observations are conditionally independent given $s$.
- Each model is queried at most once per item, since re-running the same model on the
  same input gives the same answer.

**Terminal decision.** Accepting has expected reward $bR_{TP} - (1-b)C_{FP}$ and rejecting
has $-bC_{FN}$, so it is optimal to accept iff

$$b \ge b^* = \frac{C_{FP}}{R_{TP} + C_{FN} + C_{FP}}.$$

**Escalation.** From belief $b$, querying the expensive model is worth it iff

$$\sum_{o} P(o\mid b)\,\max_{d}\,\mathbb{E}[r_d \mid b'_o] \;-\; c_{\text{exp}} \;>\; \max_d \mathbb{E}[r_d \mid b].$$

The set of beliefs where this holds is an interval $[b_{\text{low}}, b_{\text{high}}]$ that
contains $b^*$.

## Policies

| Policy | Behavior |
|---|---|
| `cheap_only` | Query cheap once, then accept iff $x \ge t$ |
| `expensive_only` | Query expensive once, then accept iff it returns positive |
| `fixed_cascade` | Query cheap. If $x < t$, reject. Otherwise query expensive and accept iff it returns positive |
| `tuned_cascade` | Same as `fixed_cascade`, with $t$ chosen to maximize reward on a separate tuning batch (model-free, needs labeled data) |
| `posterior_threshold` | Query cheap. Reject if $b \le b_{\text{low}}$, accept if $b \ge b_{\text{high}}$, otherwise query expensive and accept iff $b \ge b^*$ |

The baselines turn the score into a hard label at a fixed operating point $t$
(`decision_threshold`). The tuned cascade searches $t$ over a grid, including $\pm\infty$, on
`--tune-seed` and is evaluated on `--seed`, so the comparison has no hindsight bias. The adaptive policy uses the full posterior instead. Its belief
thresholds are equivalent to two score thresholds, reject below $x_{\text{low}}$ and accept
above $x_{\text{high}}$, whose positions shift with prevalence, costs, and model quality.

The thresholds are derived from the reward model through the value-of-information condition
above, not hand-tuned. Only one query can follow the cheap one, so this one-step lookahead
is Bayes-optimal among cheap-first policies. You can also pass thresholds by hand with
`PosteriorThreshold(low, high, accept_threshold)`.

**Structural difference from a cascade.** A cascade has one cutoff (reject or escalate). The
adaptive policy has two, so it can also accept confident cheap scores without escalating.
This is where it can strictly beat even a tuned cascade: common targets, a strong cheap model,
or an expensive model too costly to use at all.

## Budget interpretation

The budgeted problem maximizes $\mathbb{E}[\text{terminal reward}]$ subject to
$\mathbb{E}[\text{query cost}] \le B$. Its Lagrangian is
$\mathbb{E}[\text{terminal reward}] - \lambda\,\mathbb{E}[\text{cost}]$.
`PosteriorThreshold.from_params(params, cost_scale=λ)` plans as if the expensive query cost
$\lambda c_{\text{exp}}$.
- $\lambda = 1$ is the net-reward policy.
- Sweeping $\lambda$ traces a frontier of cost vs terminal reward. Each point is optimal among
  cheap-first policies for its own budget.

## Robustness

The simulator can give the agent assumed parameters that differ from the true ones.
`experiments/robustness.py` scales the assumed prevalence or the assumed cheap-model $d'$ by a
ratio $r$ and reports two quantities:
- **regret:** reward lost relative to the correctly specified policy;
- **break-even ratios:** where the misspecified policy falls below the model-free tuned cascade.

## Defaults

| | Parameters | cost |
|---|---|---|
| cheap | $d' = 1.96$, baseline threshold $t = 1.28$ (TPR 0.75, FPR 0.10 at $t$) | 1 |
| expensive | TPR 0.95, FPR 0.02 | 10 |

Other defaults: $p_{\text{target}} = 0.01$, $R_{TP}=500$, $C_{FP}=50$, $C_{FN}=500$.
All of these are set in `models.py`.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python experiments/make_paper_assets.py    # every paper figure and table (~2 min)
.venv/bin/python experiments/make_paper_assets.py --quick   # smoke test (~15 s)
```

Individual experiments:

```bash
.venv/bin/python experiments/policy_diagram.py        # Fig 1
.venv/bin/python experiments/run_default.py           # Table 1
.venv/bin/python experiments/frontier.py              # Fig 2
.venv/bin/python experiments/robustness.py            # Fig 4, Table 3
.venv/bin/python experiments/run_default.py --help   # override p_target, cheap d'/threshold, expensive cost
.venv/bin/python experiments/sweep_prevalence.py      # p_target in {0.001, ..., 0.1}
.venv/bin/python experiments/sweep_expensive_cost.py  # expensive cost in {1, ..., 100}
.venv/bin/python experiments/sweep_cheap_quality.py   # cheap d' in {0.5, ..., 4}; baseline FPR held at 10%
```

Each sweep writes a CSV and a supplementary 4-panel figure. `make_paper_assets.py` combines
the three sweeps into Fig 3 and Table 2.

| Asset | File | Content |
|---|---|---|
| Fig 1 | `figures/policy_diagram.png` | Reject / escalate / accept regions in score space |
| Table 1 | `results/default_comparison.tex` | Default setting, paired differences vs tuned cascade |
| Fig 2 | `figures/frontier.png` | Terminal reward and recall vs query cost, p = 0.01 and 0.1 |
| Fig 3 | `figures/sweeps_summary.png` | Reward vs tuned cascade and escalation rate across sweeps |
| Table 2 | `results/sweeps_summary.tex` | Numbers behind Fig 3 |
| Fig 4 | `figures/robustness.png` | Reward vs tuned cascade under misspecification |
| Table 3 | `results/robustness_breakeven.tex` | Break-even misspecification ratios |

Every figure is written twice, as `.pdf` (vector, fonts embedded as TrueType, for the paper)
and `.png` (300 dpi preview). Figures are drawn at a 6.5 in text width, so use
`\includegraphics[width=\textwidth]{figures/frontier.pdf}` to keep the fonts at their
designed size. The tables need `\usepackage{booktabs}`. In Tables 1–2, bold marks a difference
whose 95% CI excludes zero in the adaptive policy's favor. The dagger in Table 3 marks rows
without a significant advantage at $r = 1$; say this in the caption.

**Reproducibility and uncertainty.**
- Everything is reproducible from `--seed` (evaluation) and `--tune-seed` (cascade tuning).
- All policies run on the same sampled items and model outputs (common random numbers).
- Differences between policies are paired per item and reported with 95% confidence intervals.
