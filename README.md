# splita

A/B test analysis that is correct by default, informative by design, and composable by construction.

[![PyPI version](https://img.shields.io/pypi/v/splita)](https://pypi.org/project/splita/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Naareman/splita/tests.yml?label=tests)](https://github.com/Naareman/splita/actions)
[![Coverage](https://img.shields.io/codecov/c/github/Naareman/splita)](https://codecov.io/gh/Naareman/splita)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/splita/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Quick Start

```python
from splita import Experiment

result = Experiment([0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 1, 1]).run()
print(result)
```

```
ExperimentResult
────────────────────────────────────
  metric          conversion
  method          ztest
  control_n       8
  treatment_n     8
────────────────────────────────────
  control_mean    0.3750
  treatment_mean  0.6250
  lift            0.2500
  relative_lift   66.67%
────────────────────────────────────
  statistic       1.4639
  pvalue          0.1432
  ci              [-0.0849, 0.5849]
  significant     False
  alpha           0.0500
────────────────────────────────────
  effect_size     0.5236 (Cohen's h)
  power           0.2825 (post-hoc*)

  * Post-hoc power is a function of the p-value
    and does not provide additional information.
    Use SampleSize for prospective power analysis.
```

## Installation

```bash
pip install splita
```

For ML-powered variance reduction (CUPAC), install with the `ml` extra:

```bash
pip install splita[ml]  # adds scikit-learn
```

## What Can You Do?

### Run an A/B test

Pass your data and splita auto-detects the metric type, selects the right test, and returns everything you need.

```python
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.normal(25.0, 8.0, size=1000)
trt = rng.normal(26.5, 8.0, size=1000)

result = Experiment(ctrl, trt).run()
print(result.significant)   # True
print(result.relative_lift)  # ~6%
print(result.to_dict())      # JSON-serialisable dict
```

Override if you want: `metric='continuous'`, `method='bootstrap'`, `alternative='greater'`, or `alpha=0.01`. All configuration is keyword-only, so the defaults are always explicit.

### Plan your experiment

Figure out how many users you need before you start.

```python
from splita import SampleSize

# Conversion rate test: detect a 2pp lift from a 10% baseline
plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
print(plan.n_per_variant)  # 3843

# Revenue test: detect a $2 lift with $40 std dev
plan = SampleSize.for_mean(baseline_mean=25.0, baseline_std=40.0, mde=2.0)
print(plan.n_per_variant)  # 6280

# How long will it take?
plan = SampleSize.for_proportion(0.10, 0.02).duration(daily_users=1000)
print(plan.days_needed)  # 8

# Already have a fixed audience? Find the smallest effect you can detect.
mde = SampleSize.mde_for_proportion(baseline=0.10, n=5000)
print(f"{mde:.4f}")  # 0.0173
```

### Check data quality

Sample Ratio Mismatch invalidates everything. Check it first.

```python
from splita import SRMCheck

result = SRMCheck([4900, 5100]).run()
print(result.passed)   # True
print(result.message)  # "No sample ratio mismatch detected (p=0.0736)."

# Unequal splits and multi-variant
result = SRMCheck([8000, 2000], expected_fractions=[0.80, 0.20]).run()
```

### Test multiple metrics

When you test conversion, revenue, and retention in the same experiment, raw p-values lie. Correct them.

```python
from splita import MultipleCorrection

pvalues = [0.01, 0.04, 0.20]
labels = ["conversion", "revenue", "retention"]

result = MultipleCorrection(pvalues, labels=labels).run()
print(result.n_rejected)       # 2
print(result.adjusted_pvalues) # [0.03, 0.06, 0.20]
print(result.rejected)         # [True, False, False]
```

Methods: `'bh'` (Benjamini-Hochberg, default), `'bonferroni'`, `'holm'`, `'by'` (Benjamini-Yekutieli).

### Reduce variance

Lower variance means smaller required sample sizes. CUPED uses pre-experiment data to subtract noise.

```python
from splita.variance import CUPED, OutlierHandler
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)
pre = rng.normal(10, 2, size=200)
ctrl = pre[:100] + rng.normal(0, 1, 100)
trt = pre[100:] + 0.5 + rng.normal(0, 1, 100)

# Step 1: Cap outliers (thresholds from pooled data to avoid bias)
handler = OutlierHandler(method='winsorize')
ctrl, trt = handler.fit_transform(ctrl, trt)

# Step 2: CUPED adjustment
cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre[:100], pre[100:])
print(f"Variance reduction: {cuped.variance_reduction_:.0%}")  # ~75%

# Step 3: Run the test on adjusted data
result = Experiment(ctrl_adj, trt_adj).run()
```

### Monitor in real-time

The peeking problem: checking results daily inflates your false positive rate. mSPRT gives you always-valid p-values that stay correct no matter when you look.

```python
from splita import mSPRT
import numpy as np

test = mSPRT(metric='continuous', alpha=0.05)

# Day 1
ctrl_day1 = np.random.default_rng(1).normal(10, 2, size=100)
trt_day1 = np.random.default_rng(2).normal(10.5, 2, size=100)
state = test.update(ctrl_day1, trt_day1)
print(state.should_stop)          # False
print(state.always_valid_pvalue)  # still high

# Day 2 (incremental update)
ctrl_day2 = np.random.default_rng(3).normal(10, 2, size=200)
trt_day2 = np.random.default_rng(4).normal(10.5, 2, size=200)
state = test.update(ctrl_day2, trt_day2)
print(state.should_stop)          # may now be True
```

### Optimize with bandits

When you want to minimize regret instead of just measuring a difference, use Thompson Sampling to shift traffic toward the winner as data arrives.

```python
from splita import ThompsonSampler
import numpy as np

rng = np.random.default_rng(42)
true_rates = [0.05, 0.07, 0.06]  # arm 1 is best

ts = ThompsonSampler(n_arms=3, random_state=42)
for _ in range(1000):
    arm = ts.recommend()
    reward = rng.binomial(1, true_rates[arm])
    ts.update(arm, reward)

result = ts.result()
print(result.current_best_arm)  # 1
print(result.prob_best)         # [~0.01, ~0.95, ~0.04]
print(result.should_stop)       # True (expected loss below threshold)
```

## Full Example: E-commerce A/B Test

A complete workflow from planning through analysis.

```python
import numpy as np
from splita import Experiment, SampleSize, SRMCheck, MultipleCorrection

# 1. Plan: 10% baseline conversion, detect a 1.5pp lift
plan = SampleSize.for_proportion(baseline=0.10, mde=0.015, power=0.80)
print(f"Need {plan.n_per_variant} users per variant")
print(f"At 5,000 users/day: {plan.duration(5000).days_needed} days")

# 2. Simulate experiment data
rng = np.random.default_rng(42)
n = plan.n_per_variant
ctrl = rng.binomial(1, 0.10, size=n)
trt = rng.binomial(1, 0.115, size=n)  # true 1.5pp lift

# 3. SRM check first — always
srm = SRMCheck([len(ctrl), len(trt)]).run()
assert srm.passed, srm.message

# 4. Analyze primary metric
primary = Experiment(ctrl, trt).run()

# 5. Analyze secondary metrics (revenue, engagement)
rev_ctrl = rng.exponential(25, size=n)
rev_trt = rng.exponential(26, size=n)
revenue = Experiment(rev_ctrl, rev_trt).run()

# 6. Correct for multiple testing
corrected = MultipleCorrection(
    [primary.pvalue, revenue.pvalue],
    labels=["conversion", "revenue"],
).run()
print(corrected)
```

## Design Philosophy

**Correct by default.** Every formula is validated against scipy.stats. Auto-detection picks the right test for your data. Welch's t-test (not Student's), unpooled standard errors for confidence intervals, Cohen's h for proportions. You get the right answer without configuring anything.

**Informative errors.** Every `ValueError` follows a 3-part structure: what went wrong, the actual bad value, and a hint for fixing it.

```
`alpha` must be in (0, 1), got 1.5.
  Detail: alpha=1.5 means a 150% false positive rate.
  Hint: typical values are 0.05, 0.01, or 0.10.
```

**Composable.** All results are frozen dataclasses with `.to_dict()` for serialisation. Pipe `OutlierHandler` into `CUPED` into `Experiment` — each step is a clean function of its inputs. No global state, no side effects.

**Zero opinions on your data stack.** splita takes arrays and returns dataclasses. It does not care whether your data comes from pandas, Spark, BigQuery, or a CSV. No DataFrame dependencies, no plotting libraries, no ORM integrations.

## API Reference

| Class | Module | Description |
|---|---|---|
| `Experiment` | `splita` | Frequentist A/B test (z-test, t-test, Mann-Whitney, chi-square, delta method, bootstrap) |
| `SampleSize` | `splita` | Power analysis for proportions, means, and ratios; MDE calculator; duration estimator |
| `SRMCheck` | `splita` | Sample Ratio Mismatch detector (chi-square goodness-of-fit) |
| `MultipleCorrection` | `splita` | Adjust p-values for multiple comparisons (BH, Bonferroni, Holm, BY) |
| `CUPED` | `splita.variance` | Variance reduction using pre-experiment covariates |
| `CUPAC` | `splita.variance` | Variance reduction using ML-predicted covariates (requires `splita[ml]`) |
| `OutlierHandler` | `splita.variance` | Outlier capping/trimming (winsorize, trim, IQR) with pooled thresholds |
| `mSPRT` | `splita.sequential` | Always-valid sequential testing (no peeking problem) |
| `GroupSequential` | `splita.sequential` | Pre-planned group sequential testing with alpha-spending functions |
| `ThompsonSampler` | `splita.bandits` | Thompson Sampling multi-armed bandit (Bernoulli, Gaussian, Poisson) |
| `LinTS` | `splita.bandits` | Linear Thompson Sampling contextual bandit |

All result types (`ExperimentResult`, `SampleSizeResult`, `SRMResult`, `CorrectionResult`, `mSPRTResult`, `BanditResult`, etc.) are frozen dataclasses with `.to_dict()` and pretty `__repr__`.

## Requirements

- Python 3.10+
- numpy >= 1.24
- scipy >= 1.10
- scikit-learn >= 1.2 (optional, only for CUPAC -- install with `pip install splita[ml]`)

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

```bash
pip install -e ".[dev]"
pytest --cov=splita
ruff check src/ tests/
mypy src/splita/
```

## License

MIT
