# Data Quality (SRM)

Data quality checks should always come before analysis. A biased sample invalidates all downstream results.

## SRMCheck

Sample Ratio Mismatch (SRM) is the most important data quality check. If the actual ratio of users in control vs treatment differs from expected, something is wrong with randomization.

```python
from splita import SRMCheck

# Equal split: 4900 control, 5100 treatment
result = SRMCheck([4900, 5100]).run()
print(result.passed)    # True
print(result.pvalue)    # 0.0736
print(result.message)   # "No sample ratio mismatch detected (p=0.0736)."
```

### Unequal splits

```python
# 80/20 split
result = SRMCheck([8000, 2000], expected_fractions=[0.80, 0.20]).run()
print(result.passed)  # True
```

### Multi-variant

```python
# Three variants with equal allocation
result = SRMCheck([3300, 3400, 3300]).run()
print(result.passed)
```

!!! warning "SRM invalidates everything"
    If `result.passed` is `False`, do not proceed with analysis. Investigate the root cause: logging bugs, bot filtering differences, redirects, or assignment bugs.

## FlickerDetector

Detects users who were assigned to multiple variants during the experiment (flickering). This can dilute treatment effects.

```python
from splita import FlickerDetector

# user_ids: who was in the experiment
# variant_counts: how many distinct variants each user saw
user_ids = [1, 2, 3, 4, 5]
variant_counts = [1, 1, 2, 1, 3]  # users 3 and 5 flickered

detector = FlickerDetector()
result = detector.check(variant_counts)
print(result.flicker_rate)    # 0.40 (2 out of 5)
print(result.flickered_count) # 2
```

## RandomizationValidator

Checks covariate balance between control and treatment to verify that randomization worked properly.

```python
from splita import RandomizationValidator
import numpy as np

rng = np.random.default_rng(42)

# Covariates for control and treatment
ctrl_covariates = {
    "age": rng.normal(35, 10, 1000),
    "tenure_days": rng.exponential(60, 1000),
    "is_premium": rng.binomial(1, 0.20, 1000).astype(float),
}
trt_covariates = {
    "age": rng.normal(35, 10, 1000),
    "tenure_days": rng.exponential(60, 1000),
    "is_premium": rng.binomial(1, 0.20, 1000).astype(float),
}

validator = RandomizationValidator()
result = validator.check(ctrl_covariates, trt_covariates)
print(result.passed)           # True (covariates are balanced)
print(result.smd)              # standardized mean differences per covariate
print(result.omnibus_pvalue)   # chi-squared omnibus test
```

## The check() function

The top-level `check()` function runs a suite of data quality checks:

```python
from splita import check
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.normal(25, 8, 1000)
trt = rng.normal(26, 8, 1000)

result = check(ctrl, trt, n_control=1000, n_treatment=1000)
print(result)
```

## Recommended workflow

```python
from splita import SRMCheck, Experiment

# Step 1: ALWAYS check SRM first
srm = SRMCheck([len(ctrl), len(trt)]).run()
assert srm.passed, f"SRM failed: {srm.message}"

# Step 2: Only then analyze
result = Experiment(ctrl, trt).run()
```

!!! tip "Automate SRM checks"
    In production pipelines, run `SRMCheck` as a gate. If it fails, alert the team and block analysis.
