# Causal Inference

splita provides 19 causal inference classes for when standard A/B testing is not possible or sufficient. These methods handle observational data, interference, cluster randomization, and more.

## DifferenceInDifferences (DiD)

Classic two-period DiD for when you cannot randomize at the user level.

```python
from splita import DifferenceInDifferences
import numpy as np

rng = np.random.default_rng(42)

# Pre and post measurements for control and treatment groups
pre_ctrl = rng.normal(10, 2, 500)
post_ctrl = rng.normal(10.5, 2, 500)   # natural trend
pre_trt = rng.normal(10, 2, 500)
post_trt = rng.normal(12, 2, 500)      # trend + treatment effect

did = DifferenceInDifferences()
result = did.fit(pre_ctrl, post_ctrl, pre_trt, post_trt)
print(result.ate)        # ~1.0 (treatment effect net of trend)
print(result.ci)         # confidence interval
print(result.pvalue)
print(result.parallel_trends_pvalue)  # test the key assumption
```

## SyntheticControl

Construct a synthetic counterfactual from a weighted combination of donor units.

```python
from splita import SyntheticControl
import numpy as np

rng = np.random.default_rng(42)

# Time series: 20 pre-treatment periods, 10 post-treatment
n_pre, n_post = 20, 10
n_donors = 5

treated = np.concatenate([
    rng.normal(50, 2, n_pre),
    rng.normal(55, 2, n_post),  # treatment effect of ~5
])
donors = rng.normal(50, 2, (n_donors, n_pre + n_post))

sc = SyntheticControl()
result = sc.fit(treated, donors, n_pre=n_pre)
print(result.ate)          # ~5.0
print(result.weights)      # donor weights
print(result.ci)
```

## PropensityScoreMatching

For observational data where treatment assignment is non-random:

```python
from splita import PropensityScoreMatching
import numpy as np

rng = np.random.default_rng(42)
n = 2000

# Covariates
X = rng.normal(0, 1, (n, 3))

# Treatment assignment depends on covariates (confounding)
propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
treatment = rng.binomial(1, propensity)

# Outcome depends on treatment AND covariates
y = 2.0 * treatment + X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, n)

psm = PropensityScoreMatching()
result = psm.fit(y, treatment, X)
print(result.ate)     # ~2.0
print(result.ci)
print(result.n_matched)
```

## ClusterExperiment

For cluster-randomized experiments (e.g., by city, store, or school):

```python
from splita import ClusterExperiment
import numpy as np

rng = np.random.default_rng(42)

# 50 clusters, ~100 users each
n_clusters = 50
cluster_sizes = rng.integers(80, 120, n_clusters)

ctrl_clusters = [rng.normal(10, 2, s) for s in cluster_sizes[:25]]
trt_clusters = [rng.normal(11, 2, s) for s in cluster_sizes[25:]]

ctrl = np.concatenate(ctrl_clusters)
trt = np.concatenate(trt_clusters)
ctrl_ids = np.concatenate([np.full(s, i) for i, s in enumerate(cluster_sizes[:25])])
trt_ids = np.concatenate([np.full(s, i + 25) for i, s in enumerate(cluster_sizes[25:])])

ce = ClusterExperiment()
result = ce.fit(ctrl, trt, ctrl_ids, trt_ids)
print(result.ate)
print(result.ci)
print(result.icc)  # intra-cluster correlation
```

## SwitchbackExperiment

For time-based switchback designs where treatment alternates over time:

```python
from splita import SwitchbackExperiment
import numpy as np

rng = np.random.default_rng(42)

# 100 time slots, alternating treatment
n_slots = 100
treatment_mask = np.array([i % 2 for i in range(n_slots)])
outcomes = rng.normal(10, 2, n_slots) + 1.5 * treatment_mask

sb = SwitchbackExperiment()
result = sb.fit(outcomes, treatment_mask)
print(result.ate)   # ~1.5
print(result.ci)
```

## DoublyRobustEstimator

Augmented Inverse Propensity Weighting (AIPW) -- robust to misspecification of either the outcome model or the propensity model:

```python
from splita import DoublyRobustEstimator
import numpy as np

rng = np.random.default_rng(42)
n = 2000
X = rng.normal(0, 1, (n, 3))
treatment = rng.binomial(1, 0.5, n)
y = 2.0 * treatment + X[:, 0] + rng.normal(0, 1, n)

dr = DoublyRobustEstimator()
result = dr.fit(y, treatment, X)
print(result.ate)
print(result.ci)
```

## TMLE

Targeted Maximum Likelihood Estimation:

```python
from splita import TMLE
import numpy as np

rng = np.random.default_rng(42)
n = 2000
X = rng.normal(0, 1, (n, 3))
treatment = rng.binomial(1, 0.5, n)
y = 2.0 * treatment + X[:, 0] + rng.normal(0, 1, n)

tmle = TMLE()
result = tmle.fit(y, treatment, X)
print(result.ate)
print(result.ci)
```

## Other causal methods

| Class | Description | Use case |
|-------|-------------|----------|
| `SurrogateEstimator` | Short-term to long-term effect prediction | When you can't wait for the long-term outcome |
| `SurrogateIndex` | Multi-surrogate index with cross-fitting | Multiple surrogate metrics |
| `InterferenceExperiment` | Network interference with Horvitz-Thompson | Social networks, marketplaces |
| `BipartiteExperiment` | Cross-side exposure mapping | Two-sided platforms |
| `ContinuousTreatmentEffect` | Dose-response curves | Variable treatment intensity |
| `DynamicCausalEffect` | Time-varying treatment effects | Treatment effects that change over time |
| `EffectTransport` | Transport effects across populations | Generalizing from one population to another |
| `GeoExperiment` | Bayesian synthetic control for geo experiments | Marketing geo-tests |
| `InstrumentalVariables` | Two-stage least squares | Non-compliance, encouragement designs |
| `MarketplaceExperiment` | Buyer/seller randomization bias | Two-sided marketplaces |
| `MediationAnalysis` | Baron-Kenny mediation (ACME) | Understanding why a treatment works |
| `RegressionDiscontinuity` | Sharp RDD with local linear regression | Policy cutoffs (age, score thresholds) |

## When to use which

| Scenario | Recommended method |
|----------|-------------------|
| Standard A/B test with clusters | `ClusterExperiment` |
| Cannot randomize; have pre/post data | `DifferenceInDifferences` |
| Cannot randomize; have covariates | `PropensityScoreMatching` or `DoublyRobustEstimator` |
| Geo-level marketing test | `GeoExperiment` or `SyntheticControl` |
| Time-based alternating design | `SwitchbackExperiment` |
| Network effects between users | `InterferenceExperiment` |
| Need robust estimation (either model can be wrong) | `DoublyRobustEstimator` or `TMLE` |
