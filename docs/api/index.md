# API Reference

splita exposes **88 classes** across 8 modules. Every class returns frozen dataclasses with `.to_dict()` and a pretty `__repr__`.

**Type legend**: **Original** = algorithm implemented from paper equations. **Wrapper** = delegates to scipy/sklearn. **Hybrid** = original algorithm + scipy/sklearn numerical primitives.

---

## Core Analysis (24 classes)

| Class | Type | Description |
|-------|------|-------------|
| `Experiment` | Hybrid | Frequentist A/B test (z-test, t-test, Mann-Whitney, chi-square, delta method, bootstrap) |
| `BayesianExperiment` | Original | Bayesian A/B test with P(B>A), expected loss, ROPE |
| `ObjectiveBayesianExperiment` | Original | Empirical Bayes A/B test with prior learned from historical experiments |
| `QuantileExperiment` | Original | Bootstrap inference at arbitrary quantiles (median, p90, p99) |
| `SampleSize` | Hybrid | Power analysis for proportions, means, and ratios |
| `SRMCheck` | Wrapper | Sample Ratio Mismatch detector (chi-square goodness-of-fit) |
| `MultipleCorrection` | Original | p-value correction (BH, Bonferroni, Holm, BY) |
| `PermutationTest` | Original | Exact distribution-free hypothesis testing via label permutation |
| `PowerSimulation` | Wrapper | Monte Carlo power for complex designs |
| `HTEEstimator` | Wrapper | Heterogeneous treatment effects (T-learner, S-learner) |
| `CausalForest` | Hybrid | Honest T-learner with jackknife CIs |
| `TriggeredExperiment` | Wrapper | Intent-to-treat vs per-protocol analysis |
| `DilutionAnalysis` | Hybrid | Dilute triggered effect back to full ITT population |
| `InteractionTest` | Hybrid | Segment-level treatment effect heterogeneity (Cochran's Q) |
| `MultiObjectiveExperiment` | Wrapper | Pareto analysis across multiple metrics |
| `StratifiedExperiment` | Original | Neyman-style stratified inference |
| `FunnelExperiment` | Hybrid | Per-step conversion funnel analysis |
| `InterleavingExperiment` | Hybrid | Team Draft / Balanced interleaving for ranking comparison |
| `MetricDecomposition` | Wrapper | Decompose metric into additive components and test each |
| `MixedEffectsExperiment` | Original | Random-intercept model for repeated-measures experiments |
| `OECBuilder` | Hybrid | Combine multiple metrics into an Overall Evaluation Criterion |
| `OptimalProxyMetrics` | Original | Learn optimal weighted proxy for a north star metric |
| `RiskAwareDecision` | Hybrid | Multi-metric constrained decision framework |
| `SurvivalExperiment` | Hybrid | Kaplan-Meier survival estimation and log-rank test |

## Variance Reduction (14 classes)

| Class | Type | Description |
|-------|------|-------------|
| `CUPED` | Original | Pre-experiment covariate adjustment |
| `CUPAC` | Hybrid | ML-predicted covariate adjustment (cross-validated) |
| `OutlierHandler` | Hybrid | Winsorize, trim, IQR, DBSCAN clustering |
| `MultivariateCUPED` | Original | Multi-covariate CUPED extension |
| `RegressionAdjustment` | Original | Lin's OLS with HC2 robust SEs |
| `AdaptiveWinsorizer` | Original | Grid-search optimal capping thresholds |
| `DoubleML` | Hybrid | Double/debiased ML for treatment effects |
| `ClusterBootstrap` | Original | Cluster-level bootstrap for ratio metrics with within-cluster correlation |
| `InExperimentVR` | Original | Variance reduction using in-experiment control-group covariates |
| `NonstationaryAdjustment` | Hybrid | Bias-corrected ATE via time-series decomposition |
| `PostStratification` | Original | Post-experiment stratification with inverse-variance weighting |
| `PredictionPoweredInference` | Original | Augment small labeled data with ML predictions for valid inference |
| `RobustMeanEstimator` | Original | Huber M-estimator, Median of Means, Catoni estimator |
| `TrimmedMeanEstimator` | Original | Robust ATE via symmetric tail trimming |

## Sequential Testing (7 classes)

| Class | Type | Description |
|-------|------|-------------|
| `mSPRT` | Original | Always-valid p-values via mixture likelihood ratio |
| `GroupSequential` | Original | Alpha-spending boundaries (OBF, Pocock, Kim-DeMets) |
| `EValue` | Original | E-value sequential testing |
| `ConfidenceSequence` | Original | Time-uniform confidence sequences |
| `EProcess` | Original | Safe testing with e-processes (GRAPA, universal) |
| `SampleSizeReestimation` | Original | Mid-experiment sample size adjustment via conditional power |
| `YEASTSequentialTest` | Original | Tuning-free sequential test via Levy's inequality |

## Bandits (5 classes)

| Class | Type | Description |
|-------|------|-------------|
| `ThompsonSampler` | Original | Multi-armed bandit (Bernoulli, Gaussian, Poisson) |
| `LinTS` | Original | Linear Thompson Sampling contextual bandit |
| `LinUCB` | Original | Upper confidence bound contextual bandit |
| `BayesianStopping` | Original | Stopping rule evaluator for bandits |
| `OfflineEvaluator` | Hybrid | Offline policy evaluation via IPS and doubly robust estimator |

## Causal Inference (19 classes)

| Class | Type | Description |
|-------|------|-------------|
| `DifferenceInDifferences` | Hybrid | Classic two-period DiD with parallel trends check |
| `SyntheticControl` | Hybrid | Weighted donor combination via constrained optimization |
| `ClusterExperiment` | Hybrid | Cluster-robust inference with ICC |
| `SwitchbackExperiment` | Hybrid | Time-based switchback design analysis |
| `SurrogateEstimator` | Wrapper | Short-term to long-term effect prediction |
| `SurrogateIndex` | Hybrid | Multi-surrogate index with cross-fitting |
| `InterferenceExperiment` | Original | Network interference with Horvitz-Thompson estimator |
| `BipartiteExperiment` | Original | Cross-side exposure mapping for two-sided experiments |
| `ContinuousTreatmentEffect` | Original | Dose-response curve via kernel-weighted local linear regression |
| `DoublyRobustEstimator` | Hybrid | Augmented IPW (AIPW) with cross-fitting |
| `DynamicCausalEffect` | Hybrid | Time-varying treatment effects with doubly robust estimation |
| `EffectTransport` | Original | Transport treatment effects across populations via inverse odds weighting |
| `GeoExperiment` | Hybrid | Bayesian synthetic control for geo-level marketing experiments |
| `InstrumentalVariables` | Original | Two-Stage Least Squares (2SLS) for LATE estimation |
| `MarketplaceExperiment` | Hybrid | Buyer/seller randomization bias analysis for two-sided marketplaces |
| `MediationAnalysis` | Original | Baron-Kenny mediation with Sobel test for indirect effects (ACME) |
| `PropensityScoreMatching` | Original | Logistic propensity scores with nearest-neighbor matching |
| `RegressionDiscontinuity` | Original | Sharp RDD with local linear regression and IK bandwidth |
| `TMLE` | Hybrid | Targeted Maximum Likelihood Estimation for ATE |

## Diagnostics (10 classes)

| Class | Type | Description |
|-------|------|-------------|
| `NoveltyCurve` | Original | Rolling-window novelty/primacy effect detection |
| `AATest` | Wrapper | Validate randomization via simulation |
| `EffectTimeSeries` | Hybrid | Cumulative treatment effect over time |
| `MetricSensitivity` | Hybrid | Pre-experiment power estimation from historical data |
| `VarianceEstimator` | Original | Distributional analysis with A/B-specific recommendations |
| `NonStationaryDetector` | Original | CUSUM change-point detection on effect series |
| `CarryoverDetector` | Wrapper | Detect treatment effect leakage into control via pre/post comparison |
| `FlickerDetector` | Original | Detect users who switched variants mid-experiment |
| `PHackingDetector` | Hybrid | p-curve analysis for selective reporting detection |
| `RandomizationValidator` | Hybrid | Covariate balance check via SMD and chi-squared omnibus test |

## Design (6 classes)

| Class | Type | Description |
|-------|------|-------------|
| `PairwiseDesign` | Original | Mahalanobis distance matched-pair assignment |
| `AdaptiveEnrichment` | Original | Mid-experiment subgroup selection based on treatment response |
| `BayesianExperimentOptimizer` | Hybrid | Surrogate-model-based experiment parameter optimization |
| `BudgetSplitDesign` | Hybrid | Budget-split design to eliminate cannibalization bias |
| `FractionalFactorialDesign` | Hybrid | Resolution III+ fractional factorial designs for factor screening |
| `ResponseAdaptiveRandomization` | Original | Dynamic allocation probabilities favouring better-performing arms |

## Governance (3 classes)

| Class | Type | Description |
|-------|------|-------------|
| `ExperimentRegistry` | Original | In-memory experiment tracking with date filtering |
| `ConflictDetector` | Original | Overlapping experiment detection (traffic, metric, segment) |
| `GuardrailMonitor` | Wrapper | Safety metric monitoring with Bonferroni-corrected stopping rules |

## Top-level functions

| Function | Description |
|----------|-------------|
| `auto()` | Auto-select and apply the best variance reduction + analysis |
| `check()` | Run a suite of data quality checks |
| `compare()` | Compare multiple experiment results |
| `diagnose()` | Run diagnostic checks on an experiment |
| `explain()` | Generate a plain-English explanation of any result |
| `log()` | Log an experiment result |
| `load_log()` | Load previously logged results |
| `meta_analysis()` | Combine results from multiple experiments |
| `monitor()` | Real-time monitoring dashboard |
| `notify()` | Send notifications (Slack, email) |
| `power_report()` | Generate a power analysis report |
| `report()` | Generate a full experiment report |
| `serve()` | Serve splita as a REST API |
| `simulate()` | Monte Carlo power simulation |
| `to_latex_table()` | Export results to LaTeX |
| `what_if()` | What-if scenario analysis |
| `audit_trail()` | Generate an audit trail for an experiment |

## Plugin system

| Function | Description |
|----------|-------------|
| `register_method()` | Register a custom statistical test method |
| `unregister_method()` | Remove a registered method |
| `get_method()` | Retrieve a registered method |
| `list_methods()` | List all registered methods |
| `clear_methods()` | Clear all registered methods |

## Datasets

| Function | Description |
|----------|-------------|
| `load_ecommerce()` | E-commerce checkout flow experiment (5,000 users, revenue, segments) |
| `load_marketplace()` | Two-sided marketplace experiment (3,000 buyers, 800 sellers) |
| `load_subscription()` | SaaS onboarding experiment (2,000 users, survival data) |
| `load_mobile_app()` | Mobile recommendation engine experiment (4,000 users, sessions, purchases) |

## Error classes

| Class | Description |
|-------|-------------|
| `SplitaError` | Base exception for all splita errors |
| `ValidationError` | Input validation failure |
| `InsufficientDataError` | Not enough data for the requested analysis |
| `NotFittedError` | Called a method that requires fitting first |
