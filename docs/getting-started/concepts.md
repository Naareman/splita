# Core Concepts

## What is A/B testing?

A/B testing (also called split testing or randomized controlled experiments) is the practice of randomly assigning users to two or more groups and measuring the difference in a metric of interest. One group sees the current experience (control, or A), and the other sees a variation (treatment, or B).

The goal is to determine whether the observed difference is real (a true treatment effect) or just noise (random variation).

## Why splita?

Most A/B testing tools are SaaS platforms that require sending your data to a third party. splita is a Python library that runs entirely on your infrastructure:

- **Correct by default.** Auto-detection picks the right test for your data. Welch's t-test (not Student's), unpooled standard errors, Cohen's h for proportions.
- **Informative errors.** Every `ValueError` tells you what went wrong, what the bad value was, and how to fix it.
- **Composable.** Pipe `OutlierHandler` into `CUPED` into `Experiment` -- each step is a clean function of its inputs.
- **No opinions on your data stack.** splita takes arrays and returns dataclasses. NumPy in, dataclass out.

## Key terminology

### Metric types

| Term | Description | Example | splita class |
|------|-------------|---------|--------------|
| **Conversion** | Binary outcome (0 or 1) | Did the user purchase? | `Experiment(metric='conversion')` |
| **Continuous** | Real-valued outcome | Revenue per user, session duration | `Experiment(metric='continuous')` |
| **Ratio** | Numerator / denominator metric | Revenue per session | `Experiment(metric='ratio')` |

### Statistical concepts

| Term | Description |
|------|-------------|
| **p-value** | Probability of seeing a result this extreme if there were no real effect. Lower = more evidence against the null. |
| **Alpha** | Your threshold for declaring significance. Typically 0.05 (5% false positive rate). |
| **Power** | Probability of detecting a real effect when it exists. Typically 0.80 (80%). |
| **MDE** | Minimum Detectable Effect -- the smallest effect size you want to be able to detect. |
| **Confidence interval** | Range of plausible values for the true effect. A 95% CI means: if you repeated the experiment many times, 95% of the intervals would contain the true effect. |
| **Effect size** | Standardized measure of the difference (Cohen's d for means, Cohen's h for proportions). |
| **SRM** | Sample Ratio Mismatch -- when the actual split ratio differs from expected, indicating a data quality problem. |
| **CUPED** | Controlled-experiment Using Pre-Experiment Data -- a variance reduction technique. |

### Frequentist vs Bayesian

splita supports both paradigms:

| | Frequentist | Bayesian |
|---|---|---|
| **Question answered** | "Is the effect statistically significant?" | "What is the probability that B is better than A?" |
| **Key output** | p-value, confidence interval | P(B > A), expected loss, credible interval |
| **splita class** | `Experiment` | `BayesianExperiment` |
| **When to use** | Standard hypothesis testing, regulatory contexts | Decision-making under uncertainty, business contexts |

```python
# Frequentist
from splita import Experiment
result = Experiment(ctrl, trt).run()
print(result.significant)  # True/False

# Bayesian
from splita import BayesianExperiment
result = BayesianExperiment(ctrl, trt).run()
print(result.prob_treatment_better)  # 0.97
print(result.expected_loss)          # 0.001
```

## Common result fields

splita result dataclasses share common field names, but some types use domain-specific naming to reflect their statistical meaning.

### Standard fields (most result types)

| Field | Type | Meaning |
|-------|------|---------|
| `pvalue` | `float` | Fixed-horizon p-value from a frequentist test. |
| `significant` | `bool` | Whether `pvalue < alpha`. |
| `ci_lower` / `ci_upper` | `float` | Lower and upper bounds of the confidence interval. |
| `lift` | `float` | Absolute difference (treatment - control). |
| `alpha` | `float` | Significance level used. |

### Sequential testing fields (mSPRTState, mSPRTResult)

| Field | Standard equivalent | Why different |
|-------|-------------------|---------------|
| `always_valid_pvalue` | `pvalue` | This is an *always-valid* p-value, valid at any stopping time. A regular p-value assumes fixed sample size, so reusing the name would be misleading. |
| `always_valid_ci_lower` / `always_valid_ci_upper` | `ci_lower` / `ci_upper` | Always-valid confidence intervals that remain valid under continuous monitoring. |
| `should_stop` | `significant` | In sequential testing, "should stop" is the actionable decision, not just "significant". |

### Bayesian fields (BayesianResult)

| Field | Standard equivalent | Why different |
|-------|-------------------|---------------|
| `prob_b_beats_a` | (no direct equivalent) | Bayesian posterior probability, not a p-value. Values near 1.0 mean strong evidence for treatment. |
| `expected_loss_a` / `expected_loss_b` | (no direct equivalent) | Expected loss from choosing each variant. A decision-theoretic quantity with no frequentist analog. |
| `ci_lower` / `ci_upper` | same names | These are *credible* interval bounds (Bayesian), not confidence intervals, despite sharing the field names. |

### Multiple testing correction fields (CorrectionResult)

| Field | Standard equivalent | Why different |
|-------|-------------------|---------------|
| `rejected` | `significant` | After correction, "rejected" is the standard term for null hypotheses that were rejected. A metric can be "significant" in isolation but not "rejected" after correction. |
| `adjusted_pvalues` | `pvalue` | These are corrected p-values, not raw p-values. |

### Mapping guide

If you are writing generic code that processes different result types, here is a quick reference for finding the "p-value-like" and "significant-like" fields:

```python
# Example: get the p-value from any result type
def get_pvalue(result):
    if hasattr(result, 'pvalue'):
        return result.pvalue
    if hasattr(result, 'always_valid_pvalue'):
        return result.always_valid_pvalue
    if hasattr(result, 'logrank_pvalue'):
        return result.logrank_pvalue  # SurvivalResult
    if hasattr(result, 'interaction_pvalue'):
        return result.interaction_pvalue  # InteractionResult
    return None
```

### The experimentation lifecycle

1. **Plan** -- Use `SampleSize` to determine how many users you need.
2. **Check** -- Use `SRMCheck` to verify data quality before analysis.
3. **Reduce variance** -- Use `CUPED` or `OutlierHandler` to improve sensitivity.
4. **Analyze** -- Use `Experiment` or `BayesianExperiment` to measure the effect.
5. **Monitor** -- Use `mSPRT` or `GroupSequential` for real-time decisions.
6. **Explain** -- Use `explain()` and `report()` to communicate results.

splita provides tools for every step.
