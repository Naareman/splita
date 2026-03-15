# Mobile App A/B Test

A mobile app experiment: testing a new recommendation engine on engagement and purchases.

## Load the dataset

```python
from splita.datasets import load_mobile_app

data = load_mobile_app()
print(data["description"])
```

```
Mobile app A/B test: new recommendation engine.
4,000 users per group over a 14-day window.
Session count uses negative binomial (overdispersed).
Session duration uses gamma distribution (right-skewed).
Tenure affects baseline engagement (log-scaled).
Expected effects: +8% sessions, +5% duration, +17% purchases.
```

The dataset contains:

- `control` / `treatment`: total session minutes per user (14-day window)
- `control_sessions` / `treatment_sessions`: session count per user
- `control_purchases` / `treatment_purchases`: in-app purchase count per user
- `pre_control` / `pre_treatment`: sessions in 7 days before experiment
- `user_tenure_days`: days since install

## The challenge

Mobile app experiments present several challenges:

1. **Multiple correlated metrics** -- sessions, duration, purchases are all related.
2. **Overdispersed count data** -- session counts are negative binomial, not normal.
3. **Tenure effects** -- new users behave very differently from long-term users.
4. **Right-skewed durations** -- session time is gamma-distributed.

## Step 1: Check data quality

```python
from splita import SRMCheck

ctrl = data["control"]
trt = data["treatment"]

srm = SRMCheck([len(ctrl), len(trt)]).run()
assert srm.passed, srm.message
print(f"SRM: passed (p={srm.pvalue:.4f})")
```

## Step 2: Handle outliers and reduce variance

```python
from splita.variance import OutlierHandler, CUPED

# Outlier handling (session minutes can have extreme values)
handler = OutlierHandler(method='winsorize')
ctrl_clean, trt_clean = handler.fit_transform(ctrl, trt)

# CUPED with pre-experiment sessions
cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(
    ctrl_clean, trt_clean,
    data["pre_control"], data["pre_treatment"],
)
print(f"Variance reduction: {cuped.variance_reduction_:.0%}")
```

## Step 3: Analyze primary metric (session minutes)

```python
from splita import Experiment

result_duration = Experiment(ctrl_adj, trt_adj).run()
print(f"Session minutes lift: {result_duration.relative_lift}")
print(f"Significant: {result_duration.significant}")
```

## Step 4: Analyze session count

```python
result_sessions = Experiment(
    data["control_sessions"], data["treatment_sessions"]
).run()
print(f"Session count lift: {result_sessions.relative_lift}")
print(f"Significant: {result_sessions.significant}")
```

## Step 5: Analyze in-app purchases

```python
result_purchases = Experiment(
    data["control_purchases"], data["treatment_purchases"]
).run()
print(f"Purchase lift: {result_purchases.relative_lift}")
print(f"Significant: {result_purchases.significant}")
```

## Step 6: Multiple testing correction

```python
from splita import MultipleCorrection

corrected = MultipleCorrection(
    [result_duration.pvalue, result_sessions.pvalue, result_purchases.pvalue],
    labels=["session_minutes", "session_count", "purchases"],
).run()
print(f"Rejected: {corrected.rejected}")
print(f"Adjusted p-values: {corrected.adjusted_pvalues}")
```

## Step 7: Mixed-effects model for tenure groups

Account for the fact that user tenure strongly predicts engagement.

```python
from splita import MixedEffectsExperiment
import numpy as np

# Create tenure groups
tenure = data["user_tenure_days"]
tenure_groups_ctrl = np.digitize(tenure[:len(ctrl)], bins=[30, 90, 180])
tenure_groups_trt = np.digitize(tenure[:len(trt)], bins=[30, 90, 180])

me = MixedEffectsExperiment()
result_me = me.fit(ctrl, trt, tenure_groups_ctrl, tenure_groups_trt)
print(f"Mixed-effects ATE: {result_me.ate:.4f}")
print(f"CI: {result_me.ci}")
```

## Step 8: Heterogeneous treatment effects by tenure

```python
from splita import InteractionTest
import numpy as np

tenure = data["user_tenure_days"]

# Analyze new vs. established users
new_mask = tenure < 30
est_mask = tenure >= 30

# New users
ctrl_new = data["control"][:len(ctrl)][new_mask[:len(ctrl)]]
trt_new = data["treatment"][:len(trt)][new_mask[:len(trt)]]
result_new = Experiment(ctrl_new, trt_new).run()
print(f"New users: lift={result_new.relative_lift}, p={result_new.pvalue:.4f}")

# Established users
ctrl_est = data["control"][:len(ctrl)][est_mask[:len(ctrl)]]
trt_est = data["treatment"][:len(trt)][est_mask[:len(trt)]]
result_est = Experiment(ctrl_est, trt_est).run()
print(f"Established users: lift={result_est.relative_lift}, p={result_est.pvalue:.4f}")
```

## Step 9: Novelty effect detection

Mobile experiments are especially prone to novelty effects (users engage more initially because the feature is new, then usage drops).

```python
from splita import NoveltyCurve
import numpy as np

# Simulate daily effect sizes over the 14-day window
rng = np.random.default_rng(42)
daily_effects = []
for day in range(14):
    # Rough daily slice
    n_per_day = len(ctrl) // 14
    start = day * n_per_day
    end = start + n_per_day
    day_result = Experiment(ctrl[start:end], trt[start:end]).run()
    daily_effects.append(day_result.lift)

nc = NoveltyCurve()
# Check for declining treatment effect over time
```

## Step 10: Explain results

```python
from splita import explain, report

print(explain(result_duration))
print("---")
print(report(result_duration))
```

## Key takeaways for mobile app experiments

1. **Handle outliers first** -- session durations have extreme right tails.
2. **CUPED with pre-experiment sessions** -- this is the strongest predictor for mobile engagement.
3. **Correct for multiple metrics** -- session count, duration, and purchases are correlated.
4. **Check for novelty effects** -- new features get artificially inflated engagement.
5. **Segment by tenure** -- treatment effects often differ dramatically for new vs. established users.
