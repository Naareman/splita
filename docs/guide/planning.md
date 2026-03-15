# Planning (Sample Size)

Before running an experiment, determine how many users you need. Under-powered experiments waste time; over-powered experiments waste traffic.

## SampleSize

The `SampleSize` class provides static methods for power analysis.

### For proportions (conversion rates)

```python
from splita import SampleSize

# Detect a 2pp lift from a 10% baseline conversion rate
plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
print(plan.n_per_variant)  # 3843
print(plan.alpha)          # 0.05
print(plan.power)          # 0.80
```

### For means (continuous metrics)

```python
# Detect a $2 lift with $40 standard deviation
plan = SampleSize.for_mean(baseline_mean=25.0, baseline_std=40.0, mde=2.0)
print(plan.n_per_variant)  # 6280
```

### Duration estimation

How many days will the experiment take?

```python
plan = SampleSize.for_proportion(0.10, 0.02)
duration = plan.duration(daily_users=1000)
print(duration.days_needed)  # 8
```

### Minimum detectable effect

Already have a fixed audience? Find the smallest effect you can detect:

```python
mde = SampleSize.mde_for_proportion(baseline=0.10, n=5000)
print(f"{mde:.4f}")  # 0.0173
```

### Custom power and alpha

```python
# 90% power at 1% significance
plan = SampleSize.for_proportion(
    baseline=0.10,
    mde=0.02,
    power=0.90,
    alpha=0.01,
)
print(plan.n_per_variant)  # larger sample needed
```

## PowerSimulation

For complex designs where closed-form formulas do not apply, use Monte Carlo simulation:

```python
from splita import PowerSimulation
import numpy as np

def my_test(ctrl, trt):
    """Custom test function that returns a p-value."""
    from scipy.stats import mannwhitneyu
    _, p = mannwhitneyu(ctrl, trt, alternative='two-sided')
    return p

sim = PowerSimulation(
    test_fn=my_test,
    n_per_variant=500,
    n_simulations=1000,
    random_state=42,
)
```

### simulate()

The top-level `simulate()` function provides a quick interface:

```python
from splita import simulate

result = simulate(
    metric='conversion',
    baseline=0.10,
    mde=0.02,
    n_per_variant=3000,
    n_simulations=1000,
    random_state=42,
)
print(result.power)          # estimated power
print(result.false_positive_rate)  # should be ~0.05
```

## Choosing the right MDE

!!! tip "Rules of thumb"
    - **Conversion rate**: 1-2 percentage points is typical for mature products.
    - **Revenue/session duration**: 5-10% relative lift is a good starting point.
    - **Start from business impact**: "What is the smallest change worth shipping?" -- that is your MDE.

## Power analysis workflow

```python
from splita import SampleSize

# Step 1: What sample size do I need?
plan = SampleSize.for_proportion(baseline=0.10, mde=0.015, power=0.80)
print(f"Need {plan.n_per_variant} users per variant")

# Step 2: How long will that take?
duration = plan.duration(daily_users=5000)
print(f"That's {duration.days_needed} days")

# Step 3: Can I detect a smaller effect with more time?
mde = SampleSize.mde_for_proportion(baseline=0.10, n=10000)
print(f"With 10K users, MDE = {mde:.4f}")
```
