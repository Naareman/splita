# Running an A/B Test

The `Experiment` class is the central entry point for frequentist A/B test analysis. It accepts raw data, infers the metric type, selects the appropriate statistical test, and returns a frozen `ExperimentResult` dataclass.

## Basic usage

```python
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.normal(25.0, 8.0, size=1000)
trt = rng.normal(26.5, 8.0, size=1000)

result = Experiment(ctrl, trt).run()
```

## The 6 test methods

splita supports 6 statistical tests. By default (`method='auto'`), it selects the best one based on your data.

### 1. Z-test (proportions)

Used automatically for binary (0/1) data. Tests the difference between two proportions.

```python
ctrl = rng.binomial(1, 0.10, size=5000)
trt = rng.binomial(1, 0.115, size=5000)

result = Experiment(ctrl, trt, method='ztest').run()
print(result.method)       # 'ztest'
print(result.effect_size)  # Cohen's h
```

!!! note
    Auto-detection selects the z-test when all values are 0 or 1.

### 2. Welch's t-test (continuous)

The default for continuous data. Uses Welch's correction for unequal variances (not Student's t-test).

```python
ctrl = rng.normal(25.0, 8.0, size=1000)
trt = rng.normal(26.5, 8.0, size=1000)

result = Experiment(ctrl, trt, method='ttest').run()
print(result.method)       # 'ttest'
print(result.effect_size)  # Cohen's d
```

### 3. Mann-Whitney U (non-parametric)

Distribution-free test for when normality assumptions are violated. Tests whether one distribution stochastically dominates the other.

```python
# Highly skewed data
ctrl = rng.exponential(10, size=500)
trt = rng.exponential(12, size=500)

result = Experiment(ctrl, trt, method='mannwhitney').run()
print(result.method)  # 'mannwhitney'
```

### 4. Chi-square test (categorical)

Tests association between treatment assignment and a categorical outcome.

```python
ctrl = rng.binomial(1, 0.10, size=5000)
trt = rng.binomial(1, 0.115, size=5000)

result = Experiment(ctrl, trt, method='chisquare').run()
print(result.method)  # 'chisquare'
```

### 5. Delta method (ratio metrics)

For metrics defined as a ratio (e.g., revenue per session). Requires denominator arrays.

```python
ctrl_num = rng.normal(50, 10, size=1000)
ctrl_den = rng.poisson(5, size=1000).astype(float) + 1
trt_num = rng.normal(55, 10, size=1000)
trt_den = rng.poisson(5, size=1000).astype(float) + 1

result = Experiment(
    ctrl_num, trt_num,
    metric='ratio',
    method='delta',
    control_denominator=ctrl_den,
    treatment_denominator=trt_den,
).run()
print(result.method)  # 'delta'
```

### 6. Bootstrap

Non-parametric resampling-based inference. Works for any metric type and makes no distributional assumptions.

```python
result = Experiment(ctrl, trt, method='bootstrap', n_bootstrap=5000, random_state=42).run()
print(result.method)  # 'bootstrap'
```

!!! tip
    Bootstrap is slower but makes no assumptions about the data distribution. Use it when sample sizes are small or distributions are unusual.

## When to use which

| Scenario | Recommended method |
|----------|-------------------|
| Binary conversion (0/1) | `ztest` (auto-selected) |
| Continuous metric, large sample | `ttest` (auto-selected) |
| Highly skewed or non-normal data | `mannwhitney` or `bootstrap` |
| Ratio metric (revenue/session) | `delta` |
| Small sample, any distribution | `bootstrap` |
| Categorical outcome | `chisquare` |

## Configuration options

All configuration is keyword-only:

```python
result = Experiment(
    ctrl, trt,
    metric='continuous',       # 'auto', 'conversion', 'continuous', 'ratio'
    method='ttest',            # 'auto', 'ttest', 'ztest', 'mannwhitney', 'chisquare', 'delta', 'bootstrap'
    alpha=0.05,                # significance level
    alternative='two-sided',   # 'two-sided', 'greater', 'less'
).run()
```

## Understanding the result

```python
result.significant      # bool -- is p < alpha?
result.pvalue           # float -- the p-value
result.lift             # float -- absolute difference (treatment - control)
result.relative_lift    # str -- percentage lift ("6.00%")
result.ci               # tuple -- confidence interval for the difference
result.effect_size      # float -- standardized effect size
result.power            # float -- post-hoc power estimate
result.control_mean     # float
result.treatment_mean   # float
result.metric           # str -- detected metric type
result.method           # str -- test used
result.to_dict()        # dict -- JSON-serializable
```

## One-sided tests

Test whether treatment is strictly better (or worse):

```python
# Test: is treatment mean GREATER than control?
result = Experiment(ctrl, trt, alternative='greater').run()

# Test: is treatment mean LESS than control?
result = Experiment(ctrl, trt, alternative='less').run()
```

## Multiple metrics

When testing multiple metrics in the same experiment, correct for multiple comparisons:

```python
from splita import MultipleCorrection

results = [
    Experiment(ctrl_conv, trt_conv).run(),
    Experiment(ctrl_rev, trt_rev).run(),
    Experiment(ctrl_engage, trt_engage).run(),
]

corrected = MultipleCorrection(
    [r.pvalue for r in results],
    labels=["conversion", "revenue", "engagement"],
).run()
print(corrected.rejected)          # [True, False, False]
print(corrected.adjusted_pvalues)  # Benjamini-Hochberg adjusted
```
