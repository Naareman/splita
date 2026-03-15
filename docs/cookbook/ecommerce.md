# E-commerce A/B Test

A complete worked example: testing a new checkout flow in an online store.

## Load the dataset

```python
from splita.datasets import load_ecommerce

data = load_ecommerce()
print(data["description"])
```

```
E-commerce A/B test: new checkout flow vs. existing.
5,000 users per group over 28 days.
Revenue is heavy-tailed (log-normal), with weekend lift.
Segments: new (50%), returning (35%), loyal (15%).
Expected effect: +1.5pp conversion uplift (~19% relative).
```

The dataset contains:

- `control` / `treatment`: revenue per user (0 if no purchase)
- `pre_control` / `pre_treatment`: pre-experiment page views
- `timestamps`: day index (0-27)
- `user_segments`: 'new', 'returning', 'loyal'

## Step 1: Plan the experiment

```python
from splita import SampleSize

# Baseline ~8% conversion, want to detect 1.5pp lift
plan = SampleSize.for_proportion(baseline=0.08, mde=0.015, power=0.80)
print(f"Need {plan.n_per_variant} users per variant")
print(f"At 1,000 users/day: {plan.duration(1000).days_needed} days")
```

## Step 2: Check data quality

```python
from splita import SRMCheck

ctrl = data["control"]
trt = data["treatment"]

# SRM check
srm = SRMCheck([len(ctrl), len(trt)]).run()
print(f"SRM passed: {srm.passed} (p={srm.pvalue:.4f})")
assert srm.passed, srm.message
```

## Step 3: Handle outliers

Revenue data is heavy-tailed. Winsorize before analysis.

```python
from splita.variance import OutlierHandler

handler = OutlierHandler(method='winsorize')
ctrl_clean, trt_clean = handler.fit_transform(ctrl, trt)
```

## Step 4: Apply CUPED variance reduction

Use pre-experiment page views to reduce variance.

```python
from splita.variance import CUPED

cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(
    ctrl_clean, trt_clean,
    data["pre_control"], data["pre_treatment"],
)
print(f"Variance reduction: {cuped.variance_reduction_:.0%}")
```

## Step 5: Analyze the primary metric (revenue)

```python
from splita import Experiment

result = Experiment(ctrl_adj, trt_adj).run()
print(result)
```

## Step 6: Analyze conversion rate separately

```python
import numpy as np

ctrl_conv = (data["control"] > 0).astype(float)
trt_conv = (data["treatment"] > 0).astype(float)

conv_result = Experiment(ctrl_conv, trt_conv).run()
print(f"Conversion lift: {conv_result.relative_lift}")
print(f"Significant: {conv_result.significant}")
```

## Step 7: Correct for multiple testing

```python
from splita import MultipleCorrection

corrected = MultipleCorrection(
    [result.pvalue, conv_result.pvalue],
    labels=["revenue", "conversion"],
).run()
print(f"Rejected: {corrected.rejected}")
print(f"Adjusted p-values: {corrected.adjusted_pvalues}")
```

## Step 8: Segment analysis

Check if the treatment effect varies by user segment.

```python
from splita import InteractionTest

segments = data["user_segments"]
unique_segments = np.unique(segments)

segment_results = {}
for seg in unique_segments:
    mask = segments == seg
    seg_result = Experiment(
        data["control"][mask],
        data["treatment"][mask],
    ).run()
    segment_results[seg] = seg_result
    print(f"{seg}: lift={seg_result.lift:.4f}, p={seg_result.pvalue:.4f}")
```

## Step 9: Explain and report

```python
from splita import explain, report

print(explain(result))
print("---")
print(report(result))
```

## Step 10: Bayesian perspective

```python
from splita import BayesianExperiment

bayes = BayesianExperiment(ctrl_adj, trt_adj).run()
print(f"P(treatment better): {bayes.prob_treatment_better:.3f}")
print(f"Expected loss: {bayes.expected_loss:.5f}")
```

## Full pipeline in 15 lines

```python
from splita import Experiment, SRMCheck, explain
from splita.datasets import load_ecommerce
from splita.variance import CUPED, OutlierHandler

data = load_ecommerce()
ctrl, trt = data["control"], data["treatment"]

assert SRMCheck([len(ctrl), len(trt)]).run().passed
ctrl, trt = OutlierHandler(method='winsorize').fit_transform(ctrl, trt)
ctrl, trt = CUPED().fit_transform(ctrl, trt, data["pre_control"], data["pre_treatment"])

result = Experiment(ctrl, trt).run()
print(explain(result))
```
