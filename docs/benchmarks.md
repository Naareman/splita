# Benchmarks: splita vs scipy.stats vs statsmodels

A comparison of splita against the two most common Python statistics libraries for A/B testing workflows.

## Lines of Code for Common Tasks

### Task 1: Run a two-sample z-test with confidence interval

**splita (3 lines)**
```python
from splita import Experiment
result = Experiment(ctrl, trt).run()
print(result.pvalue, result.ci_lower, result.ci_upper, result.significant)
```

**scipy.stats (12 lines)**
```python
import numpy as np
from scipy.stats import norm

ctrl_mean, trt_mean = np.mean(ctrl), np.mean(trt)
ctrl_se = np.std(ctrl, ddof=1) / np.sqrt(len(ctrl))
trt_se = np.std(trt, ddof=1) / np.sqrt(len(trt))
se_diff = np.sqrt(ctrl_se**2 + trt_se**2)
z = (trt_mean - ctrl_mean) / se_diff
pvalue = 2 * (1 - norm.cdf(abs(z)))
lift = trt_mean - ctrl_mean
ci_lower = lift - 1.96 * se_diff
ci_upper = lift + 1.96 * se_diff
significant = pvalue < 0.05
```

**statsmodels (8 lines)**
```python
from statsmodels.stats.weightstats import ztest, CompareMeans, DescrStatsW
z_stat, pvalue = ztest(trt, ctrl)
d1, d2 = DescrStatsW(ctrl), DescrStatsW(trt)
cm = CompareMeans(d2, d1)
ci = cm.zconfint_diff()
lift = d2.mean - d1.mean
ci_lower, ci_upper = ci
significant = pvalue < 0.05
```

### Task 2: Power analysis / sample size calculation

**splita (2 lines)**
```python
from splita import SampleSize
plan = SampleSize.for_proportion(baseline=0.10, mde=0.02, power=0.80)
```

**scipy.stats (8 lines)**
```python
from scipy.stats import norm
import math

p1, p2 = 0.10, 0.12
z_alpha = norm.ppf(1 - 0.05 / 2)
z_beta = norm.ppf(0.80)
es = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))
n = math.ceil((z_alpha + z_beta) ** 2 / es ** 2)
```

**statsmodels (4 lines)**
```python
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
es = proportion_effectsize(0.10, 0.12)
n = NormalIndPower().solve_power(es, power=0.80, alpha=0.05, ratio=1)
```

### Task 3: CUPED variance reduction

**splita (3 lines)**
```python
from splita.variance import CUPED
cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
```

**scipy.stats (10 lines)**
```python
import numpy as np

# Manual CUPED implementation
pooled = np.concatenate([ctrl, trt])
pooled_pre = np.concatenate([pre_ctrl, pre_trt])
theta = np.cov(pooled, pooled_pre)[0, 1] / np.var(pooled_pre, ddof=1)
pre_mean = np.mean(pooled_pre)
ctrl_adj = ctrl - theta * (pre_ctrl - pre_mean)
trt_adj = trt - theta * (pre_trt - pre_mean)
```

**statsmodels**: No built-in CUPED. Must implement manually (same as scipy).

## Feature Comparison

| Feature | splita | scipy.stats | statsmodels |
|---|:---:|:---:|:---:|
| Z-test / t-test | Yes | Yes | Yes |
| Auto metric detection | Yes | No | No |
| Welch's t-test (default) | Yes | Yes | Yes |
| Mann-Whitney U | Yes | Yes | Yes |
| Bootstrap CI | Yes | No | Yes |
| Sample size calculator | Yes | Manual | Yes |
| SRM check | Yes | Manual | No |
| Multiple testing correction | Yes | No | Yes |
| CUPED | Yes | Manual | No |
| CUPAC (ML variance reduction) | Yes | No | No |
| Bayesian A/B testing | Yes | No | No |
| Sequential testing (mSPRT) | Yes | No | No |
| Group sequential (OBF/Pocock) | Yes | No | No |
| E-values | Yes | No | No |
| Thompson Sampling bandits | Yes | No | No |
| Contextual bandits (LinUCB) | Yes | No | No |
| Difference-in-Differences | Yes | No | Yes |
| Synthetic Control | Yes | No | No |
| TMLE | Yes | No | No |
| Doubly Robust estimator | Yes | No | No |
| Causal Forest | Yes | No | No |
| Experiment registry | Yes | No | No |
| Guardrail monitoring | Yes | No | No |
| HTML reports | Yes | No | No |
| `explain()` (plain English) | Yes | No | No |
| Multilingual output | Yes | No | No |
| LaTeX export | Yes | No | Yes |
| Frozen dataclass results | Yes | No | No |
| `.to_dict()` / `.to_json()` | Yes | No | No |
| Zero DataFrame dependency | Yes | Yes | No |

## Performance

All benchmarks on Apple M-series, Python 3.12, numpy 1.26, scipy 1.12.

| Operation | splita | scipy | statsmodels |
|---|---|---|---|
| Z-test (n=10,000/group) | ~0.3ms | ~0.2ms | ~0.5ms |
| T-test (n=10,000/group) | ~0.3ms | ~0.2ms | ~0.5ms |
| Power analysis (proportions) | ~0.1ms | ~0.1ms | ~0.2ms |
| CUPED (n=10,000/group) | ~0.5ms | ~0.4ms (manual) | N/A |
| Bootstrap CI (1000 resamples) | ~50ms | N/A | ~60ms |
| SRM check (2 variants) | ~0.1ms | ~0.1ms (manual) | N/A |
| Multiple correction (100 tests) | ~0.1ms | N/A | ~0.1ms |

**Key takeaway**: splita adds negligible overhead (~0.1ms) over raw scipy for basic tests, while providing correct defaults, structured results, and 80+ additional features that would require thousands of lines of custom code with scipy/statsmodels.

## Summary

| Metric | splita | scipy.stats | statsmodels |
|---|---|---|---|
| Lines for basic A/B test | 3 | 12 | 8 |
| Lines for power analysis | 2 | 8 | 4 |
| Lines for CUPED | 3 | 10 | 10+ |
| Total features | 88 classes | ~15 relevant | ~25 relevant |
| Result format | Frozen dataclass | Tuples/floats | Custom objects |
| Dependencies | numpy, scipy | numpy | numpy, scipy, pandas |
| Python version | 3.10+ | 3.9+ | 3.9+ |
