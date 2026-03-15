# Bandits

When you want to minimize regret rather than just measure a difference, use bandit algorithms. They shift traffic toward the winning variant as data arrives, reducing the cost of experimentation.

## ThompsonSampler

Multi-armed Thompson Sampling for Bernoulli, Gaussian, or Poisson rewards.

```python
from splita import ThompsonSampler
import numpy as np

rng = np.random.default_rng(42)
true_rates = [0.05, 0.07, 0.06]  # arm 1 is best

ts = ThompsonSampler(n_arms=3, random_state=42)
for _ in range(1000):
    arm = ts.recommend()
    reward = rng.binomial(1, true_rates[arm])
    ts.update(arm, reward)

result = ts.result()
print(result.current_best_arm)  # 1
print(result.prob_best)         # [~0.01, ~0.95, ~0.04]
print(result.should_stop)       # True (expected loss below threshold)
print(result.total_reward)      # cumulative reward
```

### Gaussian rewards

```python
ts = ThompsonSampler(n_arms=2, reward_type='gaussian', random_state=42)
for _ in range(500):
    arm = ts.recommend()
    reward = rng.normal([10, 12][arm], 2)
    ts.update(arm, reward)

result = ts.result()
print(result.current_best_arm)
```

## LinTS (Linear Thompson Sampling)

Contextual bandit that uses context features to personalize arm selection.

```python
from splita import LinTS
import numpy as np

rng = np.random.default_rng(42)
n_arms = 3
d = 5  # context dimension

# True weight vectors per arm
true_weights = rng.normal(0, 1, (n_arms, d))

lints = LinTS(n_arms=n_arms, n_features=d, random_state=42)

for _ in range(2000):
    context = rng.normal(0, 1, d)
    arm = lints.recommend(context)
    reward = context @ true_weights[arm] + rng.normal(0, 0.1)
    lints.update(arm, context, reward)

result = lints.result()
print(result.current_best_arm)
```

## LinUCB

Upper Confidence Bound contextual bandit. More exploitative than LinTS.

```python
from splita import LinUCB
import numpy as np

rng = np.random.default_rng(42)
n_arms = 3
d = 5

true_weights = rng.normal(0, 1, (n_arms, d))

linucb = LinUCB(n_arms=n_arms, n_features=d, alpha=1.0)

for _ in range(2000):
    context = rng.normal(0, 1, d)
    arm = linucb.recommend(context)
    reward = context @ true_weights[arm] + rng.normal(0, 0.1)
    linucb.update(arm, context, reward)

result = linucb.result()
print(result.current_best_arm)
```

## BayesianStopping

Evaluate stopping rules for bandit experiments:

```python
from splita import BayesianStopping, ThompsonSampler
import numpy as np

rng = np.random.default_rng(42)

ts = ThompsonSampler(n_arms=2, random_state=42)
for _ in range(500):
    arm = ts.recommend()
    reward = rng.binomial(1, [0.10, 0.12][arm])
    ts.update(arm, reward)

stopping = BayesianStopping()
result = stopping.evaluate(ts)
print(result.should_stop)
print(result.prob_best)
print(result.expected_remaining_loss)
```

## OfflineEvaluator

Evaluate a new policy using historical logged data (Inverse Propensity Scoring and Doubly Robust estimation).

```python
from splita import OfflineEvaluator
import numpy as np

rng = np.random.default_rng(42)
n = 5000
n_arms = 3

# Historical data
contexts = rng.normal(0, 1, (n, 5))
actions = rng.integers(0, n_arms, n)
rewards = rng.binomial(1, 0.1, n).astype(float)
propensities = np.full(n, 1.0 / n_arms)

evaluator = OfflineEvaluator()
result = evaluator.evaluate(
    contexts=contexts,
    actions=actions,
    rewards=rewards,
    propensities=propensities,
    new_policy=lambda ctx: 1,  # always pick arm 1
)
print(result.ips_estimate)
print(result.dr_estimate)
```

## A/B test vs bandit: when to use which

| Scenario | Recommendation |
|----------|---------------|
| Need a clean measurement of the effect | A/B test (`Experiment`) |
| Want to minimize regret during the test | Bandit (`ThompsonSampler`) |
| Regulatory or scientific context | A/B test |
| Personalization (different best arm per user) | Contextual bandit (`LinTS`, `LinUCB`) |
| Short-lived promotions or campaigns | Bandit |
| Need to evaluate a policy offline | `OfflineEvaluator` |
