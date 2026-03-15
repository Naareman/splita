# Bayesian Analysis

Bayesian A/B testing answers a different question than frequentist testing. Instead of "Is the result statistically significant?", it asks "What is the probability that treatment is better than control?"

## BayesianExperiment

```python
from splita import BayesianExperiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.binomial(1, 0.10, 5000)
trt = rng.binomial(1, 0.12, 5000)

result = BayesianExperiment(ctrl, trt).run()
print(result.prob_treatment_better)  # ~0.99
print(result.expected_loss)          # expected loss if you ship treatment
print(result.credible_interval)      # 95% credible interval
print(result.lift)                   # posterior mean of the difference
```

## Understanding the results

### P(B > A)

The probability that the treatment mean/rate is higher than control. This is what most business stakeholders actually want to know.

- **> 0.95**: Strong evidence for treatment
- **0.90 - 0.95**: Moderate evidence
- **< 0.90**: Insufficient evidence

### Expected loss

The expected cost of choosing the wrong variant. If you ship treatment and it turns out to be worse, how much do you lose on average?

```python
# Decision rule: ship if expected loss < threshold
if result.expected_loss < 0.001:  # less than 0.1% loss
    print("Safe to ship")
```

### ROPE (Region of Practical Equivalence)

Define a range of effects you consider practically equivalent to zero:

```python
result = BayesianExperiment(
    ctrl, trt,
    rope=(-0.005, 0.005),  # +/- 0.5pp is "practically zero"
).run()
print(result.prob_in_rope)  # probability the effect is negligible
```

## Conversion metrics

For binary (0/1) data, `BayesianExperiment` uses a Beta-Binomial model:

```python
ctrl = rng.binomial(1, 0.10, 5000)
trt = rng.binomial(1, 0.115, 5000)

result = BayesianExperiment(ctrl, trt, metric='conversion').run()
```

The default prior is Beta(1, 1) -- a uniform prior. You can specify informative priors:

```python
result = BayesianExperiment(
    ctrl, trt,
    metric='conversion',
    prior={"alpha": 10, "beta": 90},  # prior centered at 10%
).run()
```

## Continuous metrics

For continuous data, `BayesianExperiment` uses a Normal-Inverse-Gamma model:

```python
ctrl = rng.normal(25, 8, 1000)
trt = rng.normal(26.5, 8, 1000)

result = BayesianExperiment(ctrl, trt, metric='continuous').run()
print(result.prob_treatment_better)
print(result.credible_interval)
```

The default prior is vague (non-informative). Specify informative priors with:

```python
result = BayesianExperiment(
    ctrl, trt,
    metric='continuous',
    prior={"mu": 25, "kappa": 1, "alpha": 3, "beta": 100},
).run()
```

## ObjectiveBayesianExperiment

Empirical Bayes: learn the prior from historical experiment data rather than specifying it manually.

```python
from splita import ObjectiveBayesianExperiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.binomial(1, 0.10, 5000)
trt = rng.binomial(1, 0.12, 5000)

# Historical lift data from past experiments
historical_lifts = rng.normal(0.01, 0.02, 50)

result = ObjectiveBayesianExperiment(
    ctrl, trt,
    historical_effects=historical_lifts,
).run()
print(result.prob_treatment_better)
print(result.empirical_prior)
```

## Frequentist vs Bayesian: when to use which

| Scenario | Recommendation |
|----------|---------------|
| Regulatory/scientific context | Frequentist (`Experiment`) |
| Business decision-making | Bayesian (`BayesianExperiment`) |
| Need to communicate to non-statisticians | Bayesian (P(B>A) is intuitive) |
| Want to use informative priors | Bayesian |
| Need always-valid monitoring | Sequential (`mSPRT`) |
| Standard hypothesis test | Frequentist |

## Bayesian decision framework

```python
from splita import BayesianExperiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.binomial(1, 0.10, 5000)
trt = rng.binomial(1, 0.115, 5000)

result = BayesianExperiment(ctrl, trt, rope=(-0.005, 0.005)).run()

# Decision rules
if result.prob_treatment_better > 0.95 and result.expected_loss < 0.001:
    decision = "Ship treatment"
elif result.prob_in_rope > 0.90:
    decision = "No practical difference -- keep control"
else:
    decision = "Inconclusive -- collect more data"

print(decision)
```
