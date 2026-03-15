# Variance Reduction

Lower variance means smaller required sample sizes and faster experiments. splita provides 14 variance reduction methods.

## CUPED

Controlled-experiment Using Pre-Experiment Data. The most widely used variance reduction technique at companies like Microsoft, Booking.com, and Netflix.

```python
from splita.variance import CUPED
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)

# Pre-experiment data (e.g., last week's page views)
pre_ctrl = rng.normal(10, 2, size=1000)
pre_trt = rng.normal(10, 2, size=1000)

# Experiment data (correlated with pre-experiment)
ctrl = pre_ctrl + rng.normal(0, 1, 1000)
trt = pre_trt + 0.5 + rng.normal(0, 1, 1000)

# Apply CUPED
cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
print(f"Variance reduction: {cuped.variance_reduction_:.0%}")  # ~75%

# Run the test on adjusted data
result = Experiment(ctrl_adj, trt_adj).run()
```

!!! note
    CUPED works best when the pre-experiment covariate is highly correlated with the outcome. Typical reductions range from 30-80%.

## CUPAC

ML-predicted covariate adjustment. Uses cross-validated ML models to predict the outcome from covariates, then adjusts using the predictions. Requires `pip install splita[ml]`.

```python
from splita.variance import CUPAC
import numpy as np

rng = np.random.default_rng(42)
n = 2000

# Multiple covariates as a feature matrix
X_ctrl = rng.normal(0, 1, (n, 5))
X_trt = rng.normal(0, 1, (n, 5))

ctrl = X_ctrl @ rng.normal(1, 0.5, 5) + rng.normal(0, 1, n)
trt = X_trt @ rng.normal(1, 0.5, 5) + 0.5 + rng.normal(0, 1, n)

cupac = CUPAC()
ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)
print(f"Variance reduction: {cupac.variance_reduction_:.0%}")
```

## OutlierHandler

Outliers inflate variance. Handle them before analysis.

```python
from splita.variance import OutlierHandler

handler = OutlierHandler(method='winsorize')
ctrl_clean, trt_clean = handler.fit_transform(ctrl, trt)
```

Available methods:

| Method | Description |
|--------|-------------|
| `'winsorize'` | Cap extreme values at percentile thresholds |
| `'trim'` | Remove extreme values entirely |
| `'iqr'` | Cap based on IQR fences |

!!! warning
    Always fit on pooled data (both groups together) to avoid introducing bias. `fit_transform()` handles this automatically.

## MultivariateCUPED

Extension of CUPED for multiple covariates without requiring ML:

```python
from splita.variance import MultivariateCUPED
import numpy as np

rng = np.random.default_rng(42)
n = 1000

ctrl = rng.normal(25, 8, n)
trt = rng.normal(26, 8, n)

# Multiple pre-experiment covariates
pre_ctrl = rng.normal(0, 1, (n, 3))
pre_trt = rng.normal(0, 1, (n, 3))

mcuped = MultivariateCUPED()
ctrl_adj, trt_adj = mcuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
```

## AdaptiveWinsorizer

Automatically finds the optimal capping thresholds via grid search:

```python
from splita.variance import AdaptiveWinsorizer

winsorizer = AdaptiveWinsorizer()
ctrl_clean, trt_clean = winsorizer.fit_transform(ctrl, trt)
print(f"Optimal lower: {winsorizer.lower_percentile_}")
print(f"Optimal upper: {winsorizer.upper_percentile_}")
```

## RegressionAdjustment

Lin's regression adjustment with HC2 robust standard errors:

```python
from splita.variance import RegressionAdjustment
import numpy as np

rng = np.random.default_rng(42)
n = 500

ctrl = rng.normal(25, 8, n)
trt = rng.normal(26, 8, n)
x_ctrl = rng.normal(0, 1, n)
x_trt = rng.normal(0, 1, n)

ra = RegressionAdjustment()
result = ra.fit(ctrl, trt, x_ctrl, x_trt)
print(result.ate)          # adjusted treatment effect
print(result.ci)           # confidence interval
print(result.pvalue)
```

## DoubleML

Double/debiased machine learning for treatment effect estimation:

```python
from splita.variance import DoubleML
import numpy as np

rng = np.random.default_rng(42)
n = 1000

X_ctrl = rng.normal(0, 1, (n, 5))
X_trt = rng.normal(0, 1, (n, 5))
ctrl = X_ctrl @ rng.normal(1, 0.5, 5) + rng.normal(0, 1, n)
trt = X_trt @ rng.normal(1, 0.5, 5) + 0.5 + rng.normal(0, 1, n)

dml = DoubleML()
result = dml.fit(ctrl, trt, X_ctrl, X_trt)
print(result.ate)
print(result.ci)
```

## The auto() function

The top-level `auto()` function automatically selects and applies the best variance reduction strategy:

```python
from splita import auto
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.normal(25, 8, 1000)
trt = rng.normal(26, 8, 1000)
pre_ctrl = rng.normal(10, 2, 1000)
pre_trt = rng.normal(10, 2, 1000)

result = auto(ctrl, trt, pre_control=pre_ctrl, pre_treatment=pre_trt)
print(result)
```

## All 14 methods

| Class | Description | When to use |
|-------|-------------|-------------|
| `CUPED` | Pre-experiment covariate adjustment | Single covariate, most common |
| `CUPAC` | ML-predicted adjustment | Multiple covariates, non-linear relationships |
| `OutlierHandler` | Winsorize/trim/IQR outliers | Heavy-tailed metrics (revenue) |
| `MultivariateCUPED` | Multi-covariate CUPED | Multiple covariates, no ML needed |
| `RegressionAdjustment` | Lin's OLS with HC2 SEs | Linear covariate relationships |
| `AdaptiveWinsorizer` | Auto-tuned capping | When you don't know the right percentiles |
| `DoubleML` | Double/debiased ML | High-dimensional confounders |
| `ClusterBootstrap` | Cluster-level bootstrap | Within-cluster correlation |
| `InExperimentVR` | In-experiment control covariates | No pre-experiment data available |
| `NonstationaryAdjustment` | Time-series decomposition | Non-stationary treatment effects |
| `PostStratification` | Post-experiment stratification | Known population strata |
| `PredictionPoweredInference` | ML predictions + small labels | Limited labeled data |
| `RobustMeanEstimator` | Huber/Catoni/MoM estimators | Extremely heavy tails |
| `TrimmedMeanEstimator` | Symmetric tail trimming | Symmetric outlier distributions |
