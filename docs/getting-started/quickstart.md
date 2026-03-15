# Quick Start

This 5-minute tutorial walks through a complete A/B test analysis.

## 1. Create an experiment

```python
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.normal(25.0, 8.0, size=1000)
trt = rng.normal(26.5, 8.0, size=1000)

result = Experiment(ctrl, trt).run()
```

splita auto-detects that this is continuous data and selects Welch's t-test. The result is a frozen dataclass:

```python
print(result.significant)    # True
print(result.pvalue)         # < 0.05
print(result.lift)           # ~1.5
print(result.relative_lift)  # ~6%
print(result.ci)             # 95% confidence interval
print(result.effect_size)    # Cohen's d
```

## 2. Get a plain-English explanation

```python
from splita import explain

print(explain(result))
```

The `explain()` function produces a narrative summary:

```
The treatment group's mean (26.49) was higher than the control group's
mean (25.01) by 1.48 (5.9% relative lift). This difference is
statistically significant (p = 0.0001, Welch's t-test) at the 5%
significance level.
```

You can also get explanations in other languages:

```python
print(explain(result, language="ar"))  # Arabic
```

## 3. Visualize the results

```python
from splita.viz import plot_result

plot_result(result)
```

## 4. Generate a full report

```python
from splita import report

print(report(result))
```

The report includes the test summary, effect size interpretation, power analysis, and recommendations.

## 5. Export to dict or LaTeX

```python
# JSON-serializable dict
d = result.to_dict()

# LaTeX table for papers
from splita import to_latex_table
print(to_latex_table(result))
```

## 6. Try a conversion metric

```python
ctrl_conv = rng.binomial(1, 0.10, size=5000)
trt_conv = rng.binomial(1, 0.115, size=5000)

result = Experiment(ctrl_conv, trt_conv).run()
print(result.metric)          # 'conversion'
print(result.method)          # 'ztest'
print(result.relative_lift)   # ~15%
```

splita detects binary data and uses the z-test for proportions with Cohen's h as the effect size.

## 7. Add variance reduction

Pre-experiment data can reduce variance by 50-80%, letting you detect smaller effects:

```python
from splita.variance import CUPED

pre_ctrl = rng.normal(10, 2, size=1000)
pre_trt = rng.normal(10, 2, size=1000)

ctrl = pre_ctrl + rng.normal(0, 1, 1000)
trt = pre_trt + 0.5 + rng.normal(0, 1, 1000)

cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
print(f"Variance reduction: {cuped.variance_reduction_:.0%}")

result = Experiment(ctrl_adj, trt_adj).run()
print(result.significant)  # True (more power with reduced variance)
```

## Next steps

- [Core Concepts](concepts.md) -- understand the statistical foundations
- [Running an A/B Test](../guide/experiment.md) -- all 6 test methods explained
- [Planning](../guide/planning.md) -- figure out sample size before you start
- [Cookbook](../cookbook/ecommerce.md) -- full worked examples by industry
