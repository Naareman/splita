# Subscription A/B Test

A SaaS subscription experiment: testing a new onboarding flow on churn reduction.

## Load the dataset

```python
from splita.datasets import load_subscription

data = load_subscription()
print(data["description"])
```

```
SaaS subscription A/B test: new onboarding flow.
2,000 users per group observed for 90 days.
Time-to-churn follows an exponential distribution.
Plans: free (60%), basic (30%), pro (10%).
Expected effect: 15% reduction in churn hazard.
Right-censored at 90 days (users still active).
```

The dataset contains:

- `control` / `treatment`: days active before churn (or censored at 90)
- `control_churned` / `treatment_churned`: 1 if churned, 0 if still active
- `pre_control` / `pre_treatment`: feature adoption score (0-10)
- `control_plan` / `treatment_plan`: plan type ('free', 'basic', 'pro')

## The challenge

Subscription experiments involve **survival data**: some users churn during the observation window, others are still active (right-censored). Standard t-tests do not handle censoring correctly.

## Step 1: Survival analysis

```python
from splita import SurvivalExperiment

surv = SurvivalExperiment()
result = surv.fit(
    data["control"], data["treatment"],
    data["control_churned"], data["treatment_churned"],
)
print(f"Log-rank p-value: {result.pvalue:.4f}")
print(f"Significant: {result.significant}")
print(f"Median survival (control): {result.median_control:.1f} days")
print(f"Median survival (treatment): {result.median_treatment:.1f} days")
```

## Step 2: Stratified analysis by plan type

```python
from splita import StratifiedExperiment
import numpy as np

# Analyze each plan separately
plans = ['free', 'basic', 'pro']
for plan in plans:
    ctrl_mask = data["control_plan"] == plan
    trt_mask = data["treatment_plan"] == plan

    surv_plan = SurvivalExperiment()
    result_plan = surv_plan.fit(
        data["control"][ctrl_mask], data["treatment"][trt_mask],
        data["control_churned"][ctrl_mask], data["treatment_churned"][trt_mask],
    )
    n_ctrl = ctrl_mask.sum()
    n_trt = trt_mask.sum()
    print(f"{plan}: p={result_plan.pvalue:.4f}, n_ctrl={n_ctrl}, n_trt={n_trt}")
```

## Step 3: Variance reduction with CUPED

Use feature adoption score as a pre-experiment covariate for the continuous time-to-churn metric.

```python
from splita.variance import CUPED
from splita import Experiment

cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(
    data["control"], data["treatment"],
    data["pre_control"], data["pre_treatment"],
)
print(f"Variance reduction: {cuped.variance_reduction_:.0%}")

# Standard analysis on adjusted data (note: ignores censoring)
result_adj = Experiment(ctrl_adj, trt_adj).run()
print(f"Adjusted lift: {result_adj.relative_lift}")
```

## Step 4: Mixed-effects model

Account for repeated measures and plan-level variation.

```python
from splita import MixedEffectsExperiment
import numpy as np

# Plan as the grouping variable
ctrl_groups = np.array([
    {'free': 0, 'basic': 1, 'pro': 2}[p] for p in data["control_plan"]
])
trt_groups = np.array([
    {'free': 0, 'basic': 1, 'pro': 2}[p] for p in data["treatment_plan"]
])

me = MixedEffectsExperiment()
result_me = me.fit(data["control"], data["treatment"], ctrl_groups, trt_groups)
print(f"Mixed-effects ATE: {result_me.ate:.4f}")
print(f"CI: {result_me.ci}")
```

## Step 5: Bayesian analysis

```python
from splita import BayesianExperiment

bayes = BayesianExperiment(data["control"], data["treatment"]).run()
print(f"P(treatment reduces churn): {bayes.prob_treatment_better:.3f}")
print(f"Expected loss: {bayes.expected_loss:.5f}")
```

## Step 6: Sequential monitoring

In subscription experiments, data trickles in over weeks. Use sequential testing to monitor.

```python
from splita import mSPRT
import numpy as np

test = mSPRT(metric='continuous', alpha=0.05)

# Simulate weekly looks
n = len(data["control"])
batch_size = n // 4

for week in range(1, 5):
    end = week * batch_size
    state = test.update(data["control"][:end], data["treatment"][:end])
    print(f"Week {week}: should_stop={state.should_stop}, "
          f"p={state.always_valid_pvalue:.4f}")
    if state.should_stop:
        print("Early stopping: significant result detected")
        break
```

## Step 7: Explain results

```python
from splita import explain, report

print(explain(result))
print("---")
print(report(result))
```

## Key takeaways for subscription experiments

1. **Use survival analysis** (`SurvivalExperiment`) when you have censored data.
2. **Stratify by plan** -- treatment effects often vary dramatically by plan tier.
3. **CUPED with feature adoption** -- pre-experiment engagement is typically a strong predictor of churn.
4. **Sequential monitoring** -- subscription experiments run for weeks; monitor continuously with `mSPRT`.
