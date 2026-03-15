# splita v0.4.0 — Industry-Grade Methods PRD

**Date**: 2026-03-15
**Status**: TODO — Research complete, ready to build

Methods used by Uber, Microsoft, Netflix, Airbnb, Spotify, Meta, LinkedIn,
and Booking.com that we haven't covered yet. Plus features from commercial
platforms (GrowthBook, Eppo, Statsig) that no open-source Python library has.

---

## What's Missing (ranked by impact)

### Tier 1: High Impact, Used at Multiple Companies

#### 1. In-Experiment Variance Reduction (INEX)
- **Paper**: Deng, Du, Matlin, Zhang — "Variance Reduction Using In-Experiment Data" (KDD 2023)
- **Used by**: Microsoft
- **What**: CUPED uses PRE-experiment data. INEX uses IN-experiment data that is more strongly
  correlated with the outcome. Critical for sparse/delayed metrics (purchases, subscriptions)
  where pre-experiment data is weak.
- **Algorithm**: Uses control group in-experiment data as covariate (valid because control is
  unaffected by treatment). Approximate Null Augmentation (ANA) framework.
- **Why we need it**: Strictly better than CUPED for many real-world metrics.
- **Module**: `splita.variance.inex`
- **Class**: `InExperimentVarianceReduction`
- **Difficulty**: 6/10

#### 2. Metric Decomposition
- **Paper**: Deng, Hagar, Stevens, Xifara, Gandhi — "Metric Decomposition in A/B Tests" (KDD 2024)
- **Used by**: Microsoft
- **What**: Decompose a metric into components with different signal-to-noise ratios.
  Test components separately for higher sensitivity. E.g., revenue = conversion * AOV.
- **Algorithm**: Identify additive/multiplicative decomposition, test each component,
  combine with delta method.
- **Module**: `splita.core.metric_decomposition`
- **Class**: `MetricDecomposition`
- **Difficulty**: 6/10

#### 3. Interleaving Experiments
- **Paper**: Chapelle et al. (2012) + Airbnb Engineering (2024)
- **Used by**: Airbnb, Netflix, search engines
- **What**: For ranking systems — show interleaved results from two algorithms side by side.
  50x faster than standard A/B tests. Measures preference by which items users click.
- **Algorithm**: Team Draft interleaving, Balanced interleaving, Multileaving (for >2 variants).
  Credit assignment via click-through comparison.
- **Module**: `splita.core.interleaving`
- **Class**: `InterleavingExperiment`
- **Difficulty**: 5/10

#### 4. Guardrail Metric Monitor
- **Paper**: Microsoft ExP patterns (2024), Kohavi et al. (2020)
- **Used by**: Microsoft, Uber, Spotify, all major platforms
- **What**: Automated monitoring of safety/guardrail metrics during experiments.
  Auto-stops if guardrails breach thresholds. Prevents shipping features that
  improve primary metric but degrade latency, revenue, or user satisfaction.
- **Algorithm**: Sequential monitoring of multiple guardrail metrics with
  Bonferroni correction for multiple comparisons.
- **Module**: `splita.governance.guardrail`
- **Class**: `GuardrailMonitor`
- **Difficulty**: 3/10

#### 5. Overall Evaluation Criterion (OEC)
- **Paper**: Kohavi, Tang, Xu (2020) Chapter 7; Microsoft ExP
- **Used by**: Microsoft (Bing), LinkedIn
- **What**: Composite metric combining multiple signals into a single decision metric.
  Predicts long-term value from short-term proxies. Resolves the "which metric wins?" problem.
- **Algorithm**: Weighted combination of normalized metrics, validated against long-term outcomes.
- **Module**: `splita.core.oec`
- **Class**: `OECBuilder`
- **Difficulty**: 4/10

### Tier 2: Medium Impact, Important for Mature Platforms

#### 6. Dilution Analysis
- **Paper**: Deng et al. "Diluted Treatment Effect Estimation" (2015)
- **Used by**: Microsoft, Uber
- **What**: When only a fraction of users actually see the treatment (triggered analysis),
  zoom into triggered population for sensitivity, then dilute back for ship decision.
- **Algorithm**: ITT effect = triggered_effect * trigger_rate. Correct SEs via delta method.
- **Module**: Extend `splita.core.triggered`
- **Difficulty**: 4/10

#### 7. Carry-Over Effect Detection
- **Paper**: Bojinov & Shephard (2019), Uber Engineering
- **Used by**: Uber, LinkedIn
- **What**: Detects when treatment effects persist after experiment ends (contamination).
  If users in the control group are affected by the treatment group, results are biased.
- **Algorithm**: Post-experiment AA test, washout period analysis, lagged treatment effect.
- **Module**: `splita.diagnostics.carryover`
- **Class**: `CarryoverDetector`
- **Difficulty**: 5/10

#### 8. Flicker Detection
- **Paper**: Fabijan et al. (2019), Kohavi et al. (2020) Chapter 21
- **Used by**: All major platforms
- **What**: Detects users who switched between variants mid-experiment.
  Flickers invalidate results because they violate SUTVA.
- **Algorithm**: Track user-variant assignment history, flag users with multiple assignments,
  compute flicker rate, adjust analysis to exclude flickers.
- **Module**: `splita.diagnostics.flicker`
- **Class**: `FlickerDetector`
- **Difficulty**: 3/10

#### 9. Nonstationary Variance Reduction
- **Paper**: Chen, Zhang, Deng — "Nonstationary A/B Tests" (Management Science 2024)
- **Used by**: Microsoft
- **What**: Standard methods assume stationarity (constant treatment effect over time).
  Real experiments have day-of-week effects, seasonality, trending. This corrects
  both bias AND variance from nonstationarity.
- **Algorithm**: Time-series decomposition of treatment effect, bias-corrected estimator,
  heteroskedasticity-robust variance.
- **Module**: `splita.variance.nonstationary`
- **Class**: `NonstationaryAdjustment`
- **Difficulty**: 7/10

#### 10. Risk-Aware Multi-Metric Decisions
- **Paper**: Spotify Engineering (2024) — "Risk-Aware Product Decisions in A/B Tests"
- **Used by**: Spotify
- **What**: Goes beyond simple Pareto analysis. Incorporates risk preferences
  when trading off metrics. A PM can specify "I accept up to 1% conversion drop
  if revenue increases by 5%."
- **Algorithm**: Constrained optimization with user-specified tradeoff bounds,
  expected utility maximization.
- **Module**: `splita.core.risk_aware`
- **Class**: `RiskAwareDecision`
- **Difficulty**: 6/10

#### 11. Ratio Metric Cluster Bootstrap
- **Paper**: Uber Engineering, Cameron & Miller (2015)
- **Used by**: Uber
- **What**: Proper variance estimation for ratio metrics (CTR, revenue/session)
  when observations are clustered (multiple sessions per user). Standard bootstrap
  underestimates SE; cluster bootstrap is correct.
- **Algorithm**: Resample at cluster (user) level, compute ratio metric per
  bootstrap sample, percentile CI.
- **Module**: `splita.variance.cluster_bootstrap`
- **Class**: `ClusterBootstrap`
- **Difficulty**: 4/10

### Tier 3: Specialized / Cutting-Edge

#### 12. Post-Stratification Adjustment
- **Paper**: Alibaba (2024), Miratrix et al. (2013)
- **Used by**: Alibaba, Microsoft
- **What**: After data collection, stratify by observed covariates and compute
  weighted treatment effects within strata. Reduces variance without pre-registration.
- **Algorithm**: Form strata from observed covariates, compute within-stratum ATE,
  combine with variance-optimal weights.
- **Module**: `splita.variance.post_stratification`
- **Class**: `PostStratification`
- **Difficulty**: 4/10

#### 13. Robust Mean Estimators
- **Paper**: Various; Huber (1964), Catoni (2012)
- **Used by**: Various platforms internally
- **What**: Beyond winsorized/trimmed mean — Huber M-estimator, median of means,
  Catoni's estimator. More statistically efficient than simple percentile capping.
- **Algorithm**: Iteratively reweighted least squares (IRLS) for Huber M-estimator,
  median of subgroup means for median-of-means.
- **Module**: `splita.variance.robust_estimators`
- **Class**: `RobustMeanEstimator`
- **Difficulty**: 5/10

#### 14. Objective Bayesian A/B Testing
- **Paper**: Microsoft/Bing — "Objective Bayesian A/B Testing" (2019)
- **Used by**: Microsoft (Bing)
- **What**: Uses historical A/B test data to learn an objective prior (not subjective).
  Combines Bayesian and frequentist benefits — valid coverage + interpretable posteriors.
- **Algorithm**: Empirical Bayes — fit a prior from historical experiment effect sizes,
  then use that prior for new experiments.
- **Module**: `splita.core.objective_bayesian`
- **Class**: `ObjectiveBayesianExperiment`
- **Difficulty**: 6/10

#### 15. Dynamic Causal Effects with RL
- **Paper**: Shi, Deng et al. — "Dynamic Causal Effects Evaluation" (JASA 2022)
- **Used by**: Microsoft
- **What**: When treatment effects evolve over time (not static). Uses
  reinforcement learning framework to estimate time-varying causal effects.
- **Algorithm**: Doubly robust estimator with time-varying propensity scores,
  marginal structural models.
- **Module**: `splita.causal.dynamic_effects`
- **Class**: `DynamicCausalEffect`
- **Difficulty**: 8/10

---

## Features from Commercial Platforms We Should Match

| Feature | Platform | What | Priority |
|---------|----------|------|----------|
| Warehouse-native | Eppo, Statsig | Run analysis directly on data warehouse (BigQuery, Snowflake) | Out of scope (splita is a library) |
| Auto-CUPED | Statsig, GrowthBook | Automatically find best pre-period covariate | Medium |
| Metric library | All | Pre-defined metric definitions (ratio, funnel, etc.) | Low |
| Experiment lifecycle | All | Draft → Running → Analyzing → Decided states | Already have (ExperimentRegistry) |
| Power calculator UI | All | Interactive sample size planning | Out of scope (library) |
| Always-on experiments | Statsig | Experiments that never end, just accumulate evidence | Already have (mSPRT) |
| Mutual exclusion layers | All | Ensure experiments don't interfere | Extend ConflictDetector |
| Segment analysis | All | Auto-run experiment across segments | Already have (InteractionTest) |

---

## Build Order (easiest first)

| # | Method | Difficulty | Company |
|---|--------|-----------|---------|
| 1 | GuardrailMonitor | 3/10 | Microsoft, all |
| 2 | FlickerDetector | 3/10 | All |
| 3 | OECBuilder | 4/10 | Microsoft, LinkedIn |
| 4 | Dilution (extend TriggeredExperiment) | 4/10 | Microsoft, Uber |
| 5 | PostStratification | 4/10 | Alibaba |
| 6 | ClusterBootstrap | 4/10 | Uber |
| 7 | InterleavingExperiment | 5/10 | Airbnb, Netflix |
| 8 | CarryoverDetector | 5/10 | Uber, LinkedIn |
| 9 | RobustMeanEstimator | 5/10 | Various |
| 10 | InExperimentVarianceReduction | 6/10 | Microsoft |
| 11 | MetricDecomposition | 6/10 | Microsoft |
| 12 | ObjectiveBayesianExperiment | 6/10 | Microsoft (Bing) |
| 13 | RiskAwareDecision | 6/10 | Spotify |
| 14 | NonstationaryAdjustment | 7/10 | Microsoft |
| 15 | DynamicCausalEffect | 8/10 | Microsoft |

---

## Sources

- [Uber XP Platform](https://www.uber.com/blog/xp/)
- [Uber Experiment Evaluation Engine 100x Faster](https://www.uber.com/blog/making-ubers-experiment-evaluation-engine-100x-faster/)
- [Variance Reduction Using In-Experiment Data (Deng et al. KDD 2023)](https://alexdeng.github.io/public/files/kdd2023-inexp.pdf)
- [Metric Decomposition in A/B Tests (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671556)
- [Airbnb Interleaving Experiments](https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e)
- [Spotify Risk-Aware Multi-Metric Decisions (2024)](https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics)
- [Alex Deng Publications](https://alexdeng.github.io/publication/)
- [Nonstationary A/B Tests (Management Science 2024)](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.01205)
- [Microsoft ExP Patterns](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/patterns-of-trustworthy-experimentation-post-experiment-stage/)
- [Awesome Causal Inference - Industry Applications](https://github.com/matteocourthoud/awesome-causal-inference/blob/main/src/industry-applications.md)
- [Statistical Challenges in A/B Testing (American Statistician 2023)](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2257237)
- [GrowthBook Statistics](https://docs.growthbook.io/statistics/overview)
- [Alibaba Post-Stratification](https://topic.alibabacloud.com/a/ab-test-sensitivity-improvement-by-using-post-stratification_8_8_31431041.html)
- [Kohavi, Tang, Xu (2020) "Trustworthy Online Controlled Experiments"](https://exp-platform.com/)
- [Dynamic Causal Effects (JASA 2022)](https://www.tandfonline.com/doi/full/10.1080/01621459.2022.2027776)
- [Bridging Control Variates and Regression Adjustment (2025)](https://arxiv.org/html/2509.13944)
- [Fabijan et al. (2019) "Diagnosing SRM in Online Controlled Experiments"](https://dl.acm.org/doi/10.1145/3308558.3313696)
