# splita

A/B test analysis that is correct by default, informative by design, and composable by construction.

[![Tests](https://img.shields.io/github/actions/workflow/status/Naareman/splita/ci.yml?label=tests)](https://github.com/Naareman/splita/actions)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/Naareman/splita)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/splita/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://Naareman.github.io/splita/)

## Quick Start

```python
from splita import Experiment

result = Experiment([0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 1, 1]).run()
print(result)
```

```
ExperimentResult
────────────────────────────────────
  metric          conversion
  method          ztest
  control_n       8
  treatment_n     8
────────────────────────────────────
  control_mean    0.3750
  treatment_mean  0.6250
  lift            0.2500
  relative_lift   66.67%
────────────────────────────────────
  statistic       1.4639
  pvalue          0.1432
  ci              [-0.0849, 0.5849]
  significant     False
  alpha           0.0500
────────────────────────────────────
  effect_size     0.5236 (Cohen's h)
  power           0.2825 (post-hoc*)

  * Post-hoc power is a function of the p-value
    and does not provide additional information.
    Use SampleSize for prospective power analysis.
```

## Installation

```bash
pip install splita
```

For ML-powered variance reduction (CUPAC), install with the `ml` extra:

```bash
pip install splita[ml]  # adds scikit-learn
```

## What Can You Do?

### Run an A/B test

Pass your data and splita auto-detects the metric type, selects the right test, and returns everything you need.

```python
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.normal(25.0, 8.0, size=1000)
trt = rng.normal(26.5, 8.0, size=1000)

result = Experiment(ctrl, trt).run()
print(result.significant)   # True
print(result.relative_lift)  # ~6%
print(result.to_dict())      # JSON-serialisable dict
```

Override if you want: `metric='continuous'`, `method='bootstrap'`, `alternative='greater'`, or `alpha=0.01`. All configuration is keyword-only, so the defaults are always explicit.

### Plan your experiment

Figure out how many users you need before you start.

```python
from splita import SampleSize

# Conversion rate test: detect a 2pp lift from a 10% baseline
plan = SampleSize.for_proportion(baseline=0.10, mde=0.02)
print(plan.n_per_variant)  # 3843

# Revenue test: detect a $2 lift with $40 std dev
plan = SampleSize.for_mean(baseline_mean=25.0, baseline_std=40.0, mde=2.0)
print(plan.n_per_variant)  # 6280

# How long will it take?
plan = SampleSize.for_proportion(0.10, 0.02).duration(daily_users=1000)
print(plan.days_needed)  # 8

# Already have a fixed audience? Find the smallest effect you can detect.
mde = SampleSize.mde_for_proportion(baseline=0.10, n=5000)
print(f"{mde:.4f}")  # 0.0173
```

### Check data quality

Sample Ratio Mismatch invalidates everything. Check it first.

```python
from splita import SRMCheck

result = SRMCheck([4900, 5100]).run()
print(result.passed)   # True
print(result.message)  # "No sample ratio mismatch detected (p=0.0736)."

# Unequal splits and multi-variant
result = SRMCheck([8000, 2000], expected_fractions=[0.80, 0.20]).run()
```

### Test multiple metrics

When you test conversion, revenue, and retention in the same experiment, raw p-values lie. Correct them.

```python
from splita import MultipleCorrection

pvalues = [0.01, 0.04, 0.20]
labels = ["conversion", "revenue", "retention"]

result = MultipleCorrection(pvalues, labels=labels).run()
print(result.n_rejected)       # 2
print(result.adjusted_pvalues) # [0.03, 0.06, 0.20]
print(result.rejected)         # [True, False, False]
```

Methods: `'bh'` (Benjamini-Hochberg, default), `'bonferroni'`, `'holm'`, `'by'` (Benjamini-Yekutieli).

### Reduce variance

Lower variance means smaller required sample sizes. CUPED uses pre-experiment data to subtract noise.

```python
from splita.variance import CUPED, OutlierHandler
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)
pre = rng.normal(10, 2, size=200)
ctrl = pre[:100] + rng.normal(0, 1, 100)
trt = pre[100:] + 0.5 + rng.normal(0, 1, 100)

# Step 1: Cap outliers (thresholds from pooled data to avoid bias)
handler = OutlierHandler(method='winsorize')
ctrl, trt = handler.fit_transform(ctrl, trt)

# Step 2: CUPED adjustment
cuped = CUPED()
ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre[:100], pre[100:])
print(f"Variance reduction: {cuped.variance_reduction_:.0%}")  # ~75%

# Step 3: Run the test on adjusted data
result = Experiment(ctrl_adj, trt_adj).run()
```

### Monitor in real-time

The peeking problem: checking results daily inflates your false positive rate. mSPRT gives you always-valid p-values that stay correct no matter when you look.

```python
from splita import mSPRT
import numpy as np

test = mSPRT(metric='continuous', alpha=0.05)

# Day 1
ctrl_day1 = np.random.default_rng(1).normal(10, 2, size=100)
trt_day1 = np.random.default_rng(2).normal(10.5, 2, size=100)
state = test.update(ctrl_day1, trt_day1)
print(state.should_stop)          # False
print(state.always_valid_pvalue)  # still high

# Day 2 (incremental update)
ctrl_day2 = np.random.default_rng(3).normal(10, 2, size=200)
trt_day2 = np.random.default_rng(4).normal(10.5, 2, size=200)
state = test.update(ctrl_day2, trt_day2)
print(state.should_stop)          # may now be True
```

### Optimize with bandits

When you want to minimize regret instead of just measuring a difference, use Thompson Sampling to shift traffic toward the winner as data arrives.

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
```

## Full Example: E-commerce A/B Test

A complete workflow from planning through analysis.

```python
import numpy as np
from splita import Experiment, SampleSize, SRMCheck, MultipleCorrection

# 1. Plan: 10% baseline conversion, detect a 1.5pp lift
plan = SampleSize.for_proportion(baseline=0.10, mde=0.015, power=0.80)
print(f"Need {plan.n_per_variant} users per variant")
print(f"At 5,000 users/day: {plan.duration(5000).days_needed} days")

# 2. Simulate experiment data
rng = np.random.default_rng(42)
n = plan.n_per_variant
ctrl = rng.binomial(1, 0.10, size=n)
trt = rng.binomial(1, 0.115, size=n)  # true 1.5pp lift

# 3. SRM check first — always
srm = SRMCheck([len(ctrl), len(trt)]).run()
assert srm.passed, srm.message

# 4. Analyze primary metric
primary = Experiment(ctrl, trt).run()

# 5. Analyze secondary metrics (revenue, engagement)
rev_ctrl = rng.exponential(25, size=n)
rev_trt = rng.exponential(26, size=n)
revenue = Experiment(rev_ctrl, rev_trt).run()

# 6. Correct for multiple testing
corrected = MultipleCorrection(
    [primary.pvalue, revenue.pvalue],
    labels=["conversion", "revenue"],
).run()
print(corrected)
```

## Design Philosophy

**Correct by default.** Every formula is validated against scipy.stats. Auto-detection picks the right test for your data. Welch's t-test (not Student's), unpooled standard errors for confidence intervals, Cohen's h for proportions. You get the right answer without configuring anything.

**Informative errors.** Every `ValueError` follows a 3-part structure: what went wrong, the actual bad value, and a hint for fixing it.

```
`alpha` must be in (0, 1), got 1.5.
  Detail: alpha=1.5 means a 150% false positive rate.
  Hint: typical values are 0.05, 0.01, or 0.10.
```

**Composable.** All results are frozen dataclasses with `.to_dict()` for serialisation. Pipe `OutlierHandler` into `CUPED` into `Experiment` — each step is a clean function of its inputs. No global state, no side effects.

**Zero opinions on your data stack.** splita takes arrays and returns dataclasses. It does not care whether your data comes from pandas, Spark, BigQuery, or a CSV. No DataFrame dependencies, no plotting libraries, no ORM integrations.

## API Reference

88 classes across 8 modules. Every class returns frozen dataclasses with `.to_dict()`.

### Core Analysis (24 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `Experiment` | Hybrid | Frequentist A/B test (z-test, t-test, Mann-Whitney, chi-square, delta method, bootstrap) | Welch 1947, Deng et al. 2018 |
| `BayesianExperiment` | Original | Bayesian A/B test with P(B>A), expected loss, ROPE | Berry 2006 |
| `ObjectiveBayesianExperiment` | Original | Empirical Bayes A/B test with prior learned from historical experiments | Microsoft/Bing 2019 |
| `QuantileExperiment` | Original | Bootstrap inference at arbitrary quantiles (median, p90, p99) | Efron 1979 |
| `SampleSize` | Hybrid | Power analysis for proportions, means, and ratios | Farrington & Manning 1990, Cohen 1988 |
| `SRMCheck` | Wrapper | Sample Ratio Mismatch detector (chi-square goodness-of-fit) | Fabijan et al. 2019 |
| `MultipleCorrection` | Original | p-value correction (BH, Bonferroni, Holm, BY) | Benjamini & Hochberg 1995 |
| `PermutationTest` | Original | Exact distribution-free hypothesis testing via label permutation | — |
| `PowerSimulation` | Wrapper | Monte Carlo power for complex designs | — |
| `HTEEstimator` | Wrapper | Heterogeneous treatment effects (T-learner, S-learner) | Kunzel et al. 2019 |
| `CausalForest` | Hybrid | Honest T-learner with jackknife CIs | Athey & Wager 2018 |
| `TriggeredExperiment` | Wrapper | Intent-to-treat vs per-protocol analysis | Hernan & Robins 2020 |
| `DilutionAnalysis` | Hybrid | Dilute triggered effect back to full ITT population | Deng et al. 2015 |
| `InteractionTest` | Hybrid | Segment-level treatment effect heterogeneity (Cochran's Q) | Cochran 1954 |
| `MultiObjectiveExperiment` | Wrapper | Pareto analysis across multiple metrics | — |
| `StratifiedExperiment` | Original | Neyman-style stratified inference | Neyman 1923, Miratrix et al. 2013 |
| `FunnelExperiment` | Hybrid | Per-step conversion funnel analysis | Kohavi et al. 2020 |
| `InterleavingExperiment` | Hybrid | Team Draft / Balanced interleaving for ranking comparison | Chapelle et al. 2012 |
| `MetricDecomposition` | Wrapper | Decompose metric into additive components and test each | Deng et al. KDD 2024 |
| `MixedEffectsExperiment` | Original | Random-intercept model for repeated-measures experiments | — |
| `OECBuilder` | Hybrid | Combine multiple metrics into an Overall Evaluation Criterion | Deng et al. WSDM 2013 |
| `OptimalProxyMetrics` | Original | Learn optimal weighted proxy for a north star metric | Jeunen 2024 |
| `RiskAwareDecision` | Hybrid | Multi-metric constrained decision framework | Spotify 2024 |
| `SurvivalExperiment` | Hybrid | Kaplan-Meier survival estimation and log-rank test | Kaplan & Meier 1958, Cox 1972 |

### Variance Reduction (14 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `CUPED` | Original | Pre-experiment covariate adjustment | Deng et al. WSDM 2013 |
| `CUPAC` | Hybrid | ML-predicted covariate adjustment (cross-validated) | Tang et al. 2020 (DoorDash) |
| `OutlierHandler` | Hybrid | Winsorize, trim, IQR, DBSCAN clustering | Tukey 1977, Ester et al. 1996 |
| `MultivariateCUPED` | Original | Multi-covariate CUPED extension | Deng & Shi 2016 |
| `RegressionAdjustment` | Original | Lin's OLS with HC2 robust SEs | Lin 2013 |
| `AdaptiveWinsorizer` | Original | Grid-search optimal capping thresholds | Gupta et al. 2019 (Microsoft ExP) |
| `DoubleML` | Hybrid | Double/debiased ML for treatment effects | Chernozhukov et al. 2018 |
| `ClusterBootstrap` | Original | Cluster-level bootstrap for ratio metrics with within-cluster correlation | Cameron et al. 2008 |
| `InExperimentVR` | Original | Variance reduction using in-experiment control-group covariates | Deng et al. KDD 2023 |
| `NonstationaryAdjustment` | Hybrid | Bias-corrected ATE via time-series decomposition | Chen et al. 2024 |
| `PostStratification` | Original | Post-experiment stratification with inverse-variance weighting | Miratrix et al. 2013 |
| `PredictionPoweredInference` | Original | Augment small labeled data with ML predictions for valid inference | Angelopoulos et al. 2023 |
| `RobustMeanEstimator` | Original | Huber M-estimator, Median of Means, Catoni estimator | Huber 1964 |
| `TrimmedMeanEstimator` | Original | Robust ATE via symmetric tail trimming | — |

### Sequential Testing (7 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `mSPRT` | Original | Always-valid p-values via mixture likelihood ratio | Johari et al. 2015/2022 |
| `GroupSequential` | Original | Alpha-spending boundaries (OBF, Pocock, Kim-DeMets) | O'Brien & Fleming 1979, Lan & DeMets 1983 |
| `EValue` | Original | E-value sequential testing | Grunwald et al. 2020 |
| `ConfidenceSequence` | Original | Time-uniform confidence sequences | Howard et al. 2021 |
| `EProcess` | Original | Safe testing with e-processes (GRAPA, universal) | Grunwald et al. 2020, Ramdas et al. 2023 |
| `SampleSizeReestimation` | Original | Mid-experiment sample size adjustment via conditional power | Mehta & Pocock 2011 |
| `YEASTSequentialTest` | Original | Tuning-free sequential test via Levy's inequality | Kurennoy (Meta) 2024 |

### Bandits (5 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `ThompsonSampler` | Original | Multi-armed bandit (Bernoulli, Gaussian, Poisson) | Russo et al. 2018 |
| `LinTS` | Original | Linear Thompson Sampling contextual bandit | Agrawal & Goyal 2013 |
| `LinUCB` | Original | Upper confidence bound contextual bandit | Li et al. 2010 |
| `BayesianStopping` | Original | Stopping rule evaluator for bandits | — |
| `OfflineEvaluator` | Hybrid | Offline policy evaluation via IPS and doubly robust estimator | Li et al. 2011 |

### Causal Inference (19 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `DifferenceInDifferences` | Hybrid | Classic two-period DiD with parallel trends check | Card & Krueger 1994 |
| `SyntheticControl` | Hybrid | Weighted donor combination via constrained optimization | Abadie et al. 2010 |
| `ClusterExperiment` | Hybrid | Cluster-robust inference with ICC | Cameron & Miller 2015 |
| `SwitchbackExperiment` | Hybrid | Time-based switchback design analysis | Bojinov & Shephard 2019 |
| `SurrogateEstimator` | Wrapper | Short-term to long-term effect prediction | Athey et al. 2019 |
| `SurrogateIndex` | Hybrid | Multi-surrogate index with cross-fitting | Athey et al. 2019 |
| `InterferenceExperiment` | Original | Network interference with Horvitz-Thompson estimator | Basse & Feller 2018 |
| `BipartiteExperiment` | Original | Cross-side exposure mapping for two-sided experiments | Harshaw et al. 2023 |
| `ContinuousTreatmentEffect` | Original | Dose-response curve via kernel-weighted local linear regression | Hirano & Imbens 2004 |
| `DoublyRobustEstimator` | Hybrid | Augmented IPW (AIPW) with cross-fitting | Robins 1994, Bang & Robins 2005 |
| `DynamicCausalEffect` | Hybrid | Time-varying treatment effects with doubly robust estimation | Shi, Deng et al. JASA 2022 |
| `EffectTransport` | Original | Transport treatment effects across populations via inverse odds weighting | Rosenman et al. 2025 |
| `GeoExperiment` | Hybrid | Bayesian synthetic control for geo-level marketing experiments | — |
| `InstrumentalVariables` | Original | Two-Stage Least Squares (2SLS) for LATE estimation | — |
| `MarketplaceExperiment` | Hybrid | Buyer/seller randomization bias analysis for two-sided marketplaces | Bajari et al. 2023 |
| `MediationAnalysis` | Original | Baron-Kenny mediation with Sobel test for indirect effects (ACME) | Imai et al. 2010 |
| `PropensityScoreMatching` | Original | Logistic propensity scores with nearest-neighbor matching | Rosenbaum & Rubin 1983 |
| `RegressionDiscontinuity` | Original | Sharp RDD with local linear regression and IK bandwidth | Imbens & Kalyanaraman 2012 |
| `TMLE` | Hybrid | Targeted Maximum Likelihood Estimation for ATE | van der Laan & Rubin 2006 |

### Diagnostics (10 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `NoveltyCurve` | Original | Rolling-window novelty/primacy effect detection | — |
| `AATest` | Wrapper | Validate randomization via simulation | Kohavi et al. 2020 |
| `EffectTimeSeries` | Hybrid | Cumulative treatment effect over time | Kohavi et al. 2020 |
| `MetricSensitivity` | Hybrid | Pre-experiment power estimation from historical data | — |
| `VarianceEstimator` | Original | Distributional analysis with A/B-specific recommendations | — |
| `NonStationaryDetector` | Original | CUSUM change-point detection on effect series | Page 1954 |
| `CarryoverDetector` | Wrapper | Detect treatment effect leakage into control via pre/post comparison | — |
| `FlickerDetector` | Original | Detect users who switched variants mid-experiment | — |
| `PHackingDetector` | Hybrid | p-curve analysis for selective reporting detection | Simonsohn et al. 2014 |
| `RandomizationValidator` | Hybrid | Covariate balance check via SMD and chi-squared omnibus test | Austin 2009 |

### Design (6 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `PairwiseDesign` | Original | Mahalanobis distance matched-pair assignment | Greevy et al. 2004 |
| `AdaptiveEnrichment` | Original | Mid-experiment subgroup selection based on treatment response | Simon & Simon 2013 |
| `BayesianExperimentOptimizer` | Hybrid | Surrogate-model-based experiment parameter optimization | Meta 2025 |
| `BudgetSplitDesign` | Hybrid | Budget-split design to eliminate cannibalization bias | Liu et al. KDD 2021 |
| `FractionalFactorialDesign` | Hybrid | Resolution III+ fractional factorial designs for factor screening | Box & Hunter 1961 |
| `ResponseAdaptiveRandomization` | Original | Dynamic allocation probabilities favouring better-performing arms | — |

### Governance (3 classes)

| Class | Type | Description | Reference |
|---|---|---|---|
| `ExperimentRegistry` | Original | In-memory experiment tracking with date filtering | — |
| `ConflictDetector` | Original | Overlapping experiment detection (traffic, metric, segment) | Kohavi et al. 2020 |
| `GuardrailMonitor` | Wrapper | Safety metric monitoring with Bonferroni-corrected stopping rules | — |

**Type legend**: **Original** = algorithm from paper equations. **Wrapper** = delegates to scipy/sklearn. **Hybrid** = original algorithm + scipy/sklearn numerical primitives.

For full citations, see [References](https://naareman.github.io/splita/references/).

All result types are frozen dataclasses with `.to_dict()` and pretty `__repr__`.

## Why splita?

| | splita | GrowthBook | Eppo | statsmodels |
|---|---|---|---|---|
| **Type** | Python library | SaaS + SDK | SaaS + SDK | Python library |
| **Pricing** | Free (MIT) | Free tier / paid | Paid | Free (BSD) |
| **Self-hosted** | Yes (it's a library) | Yes (complex) | No | Yes (it's a library) |
| **Bayesian A/B** | Built-in | Built-in | Built-in | No |
| **Sequential testing** | mSPRT, YEAST, e-values, group sequential | CUPED + sequential | Sequential | No |
| **Variance reduction** | CUPED, CUPAC, Double ML, 14 methods | CUPED | CUPED | Manual |
| **Causal inference** | DiD, Synthetic Control, TMLE, 19 classes | No | No | DiD (basic) |
| **Bandits** | Thompson, LinUCB, LinTS | Multi-armed bandits | Bandits | No |
| **Vendor lock-in** | None | Moderate | High | None |
| **Data stays local** | Always | Self-hosted option | No | Always |
| **Jupyter-native** | Yes (HTML repr, widgets) | No | No | Partial |
| **Result format** | Frozen dataclasses, `.to_dict()` | JSON API | JSON API | Mixed |
| **Dependencies** | numpy + scipy only | Node.js / Docker | SaaS | numpy + scipy + pandas |
| **Lines for a z-test** | 3 | ~20 (SDK + config) | ~20 (SDK + config) | 8 |
| **`explain()` in 4 languages** | Yes | No | No | No |
| **LaTeX export** | Yes | No | No | Yes |

**When to use splita**: You want a Python-native experimentation toolkit with correct statistical defaults, no vendor lock-in, and your data stays in your infrastructure. You need more than basic t-tests: causal inference, sequential testing, bandits, variance reduction.

**When to use GrowthBook/Eppo**: You need a full-stack feature flagging + experimentation platform with a web UI, team collaboration, and you're okay with SaaS or self-hosted infrastructure.

**When to use statsmodels**: You need general-purpose statistics (regression, time series, GLMs) beyond A/B testing. For pure experimentation, splita provides more features with fewer lines of code.

See [Benchmarks](docs/benchmarks.md) for detailed performance comparisons.

## Requirements

- Python 3.10+
- numpy >= 1.24
- scipy >= 1.10
- scikit-learn >= 1.2 (optional, only for CUPAC -- install with `pip install splita[ml]`)

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

```bash
pip install -e ".[dev]"
pytest --cov=splita
ruff check src/ tests/
mypy src/splita/
```

## Roadmap

**Experimentation Accelerator** (planned): AI-powered experiment prioritization using content embeddings and historical A/B results to suggest which variants to test next and explain why winners win. Based on recent research in content-aware ranking for experimentation (arxiv:2602.13852, 2026). Requires LLM/embedding infrastructure — not yet in scope for a pure statistics library.

## License

MIT
