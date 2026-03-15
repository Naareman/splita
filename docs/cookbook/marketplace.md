# Marketplace A/B Test

A two-sided marketplace experiment: testing a new search ranking algorithm that affects both buyers and sellers.

## Load the dataset

```python
from splita.datasets import load_marketplace

data = load_marketplace()
print(data["description"])
```

```
Two-sided marketplace A/B test: new search ranking algorithm.
3,000 buyers and 800 sellers per group.
Buyer spend follows a Pareto distribution (heavy right tail).
Seller listings are count data (Poisson).
Expected effect: +2pp buyer conversion, +15% seller listings.
```

## The challenge

Marketplace experiments are tricky because:

1. **Heavy tails**: Buyer spend follows a power-law distribution.
2. **Two sides**: You need to measure effects on both buyers and sellers.
3. **Interference**: Buyer behavior affects seller outcomes and vice versa.

## Step 1: Analyze buyer spend

```python
from splita import Experiment, SRMCheck
from splita.variance import OutlierHandler
import numpy as np

ctrl = data["buyer_control"]
trt = data["buyer_treatment"]

# SRM check
srm = SRMCheck([len(ctrl), len(trt)]).run()
assert srm.passed

# Handle extreme outliers (Pareto distribution has very long tails)
handler = OutlierHandler(method='winsorize')
ctrl_clean, trt_clean = handler.fit_transform(ctrl, trt)

# Test buyer spend
buyer_result = Experiment(ctrl_clean, trt_clean).run()
print(f"Buyer spend lift: {buyer_result.relative_lift}")
print(f"Significant: {buyer_result.significant}")
```

## Step 2: Analyze buyer conversion

```python
ctrl_conv = (data["buyer_control"] > 0).astype(float)
trt_conv = (data["buyer_treatment"] > 0).astype(float)

conv_result = Experiment(ctrl_conv, trt_conv).run()
print(f"Buyer conversion lift: {conv_result.relative_lift}")
print(f"Significant: {conv_result.significant}")
```

## Step 3: Analyze seller listings

```python
seller_ctrl = data["seller_control"]
seller_trt = data["seller_treatment"]

seller_result = Experiment(seller_ctrl, seller_trt).run()
print(f"Seller listings lift: {seller_result.relative_lift}")
print(f"Significant: {seller_result.significant}")
```

## Step 4: Correct for multiple comparisons

```python
from splita import MultipleCorrection

corrected = MultipleCorrection(
    [buyer_result.pvalue, conv_result.pvalue, seller_result.pvalue],
    labels=["buyer_spend", "buyer_conversion", "seller_listings"],
).run()
print(f"Rejected: {corrected.rejected}")
print(f"Adjusted p-values: {corrected.adjusted_pvalues}")
```

## Step 5: Cluster-level analysis

If randomization was at the city or region level, use cluster-robust inference.

```python
from splita import ClusterExperiment
import numpy as np

rng = np.random.default_rng(42)

# Simulate cluster IDs (e.g., 30 cities per group)
n_ctrl = len(data["buyer_control"])
n_trt = len(data["buyer_treatment"])
ctrl_clusters = rng.integers(0, 30, n_ctrl)
trt_clusters = rng.integers(30, 60, n_trt)

cluster = ClusterExperiment()
cluster_result = cluster.fit(
    data["buyer_control"], data["buyer_treatment"],
    ctrl_clusters, trt_clusters,
)
print(f"Cluster-robust ATE: {cluster_result.ate:.4f}")
print(f"ICC: {cluster_result.icc:.4f}")
print(f"CI: {cluster_result.ci}")
```

## Step 6: Marketplace interference analysis

Check whether buyer-side randomization leaks into seller outcomes.

```python
from splita import MarketplaceExperiment
import numpy as np

rng = np.random.default_rng(42)

# Buyer-seller interaction matrix (which buyers transacted with which sellers)
n_buyers = len(data["buyer_control"])
n_sellers = len(data["seller_control"])

mp = MarketplaceExperiment()
# Analyze cross-side effects
```

## Step 7: Revenue per session (ratio metric)

```python
from splita import Experiment

# Revenue per session using the delta method
buyer_result_ratio = Experiment(
    data["buyer_control"], data["buyer_treatment"],
    metric='ratio',
    method='delta',
    control_denominator=data["buyer_sessions"],
    treatment_denominator=data["buyer_sessions"],
).run()
print(f"Revenue/session lift: {buyer_result_ratio.relative_lift}")
```

## Step 8: Bayesian analysis for decision-making

```python
from splita import BayesianExperiment

bayes = BayesianExperiment(ctrl_clean, trt_clean).run()
print(f"P(treatment better): {bayes.prob_treatment_better:.3f}")
print(f"Expected loss: {bayes.expected_loss:.5f}")

# Decision: ship if P(better) > 0.95 AND expected loss < threshold
if bayes.prob_treatment_better > 0.95 and bayes.expected_loss < 0.5:
    print("Ship the new ranking algorithm")
```

## Step 9: Explain results

```python
from splita import explain, report

print(explain(buyer_result))
print("---")
print(explain(seller_result))
```
