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

---

## Additional Methods Found in Research (v0.5+ backlog)

### Quasi-Experimental Methods

#### 16. Geo Experiments (GeoLift)
- **Paper**: Google GeoexperimentsResearch (open-source), Eppo GeoLift
- **Used by**: Google, Wayfair, Meta
- **What**: When you can't randomize users (ads, offline), randomize geographic regions.
  Uses Bayesian Synthetic Control at geo level. Measures incrementality of marketing spend.
- **Algorithm**: Time-based regression (TBR) or geo-based regression (GBR) on
  control vs treatment regions. Synthetic control for counterfactual.
- **Module**: `splita.causal.geo_experiment`
- **Class**: `GeoExperiment`
- **Difficulty**: 7/10

#### 17. Regression Discontinuity Design (RDD)
- **Paper**: Imbens & Lemieux (2008), Lee & Lemieux (2010)
- **Used by**: Various (policy, pricing thresholds)
- **What**: When treatment is assigned by a threshold (e.g., users above X score
  get the feature). Estimates local treatment effect at the cutoff.
- **Algorithm**: Local linear regression on both sides of the cutoff,
  bandwidth selection via MSE-optimal methods (Calonico, Cattaneo & Titiunik 2014).
- **Module**: `splita.causal.rdd`
- **Class**: `RegressionDiscontinuity`
- **Difficulty**: 6/10

#### 18. Instrumental Variables (IV)
- **Paper**: Angrist & Pischke (2009)
- **Used by**: Various (when randomization is imperfect)
- **What**: When compliance is imperfect (some users assigned to treatment don't
  actually get treated). IV estimates the Local Average Treatment Effect (LATE)
  for compliers using 2SLS.
- **Algorithm**: Two-stage least squares. First stage: predict treatment from
  instrument. Second stage: regress outcome on predicted treatment.
- **Module**: `splita.causal.instrumental_variables`
- **Class**: `InstrumentalVariables`
- **Difficulty**: 5/10

#### 19. Propensity Score Matching
- **Paper**: Rosenbaum & Rubin (1983)
- **Used by**: Various (observational causal inference)
- **What**: Match treated and control units with similar propensity scores.
  Creates pseudo-randomization from observational data.
- **Algorithm**: Logistic regression for propensity, nearest-neighbor or
  caliper matching, ATT estimation with matched pairs.
- **Module**: `splita.causal.propensity_matching`
- **Class**: `PropensityScoreMatching`
- **Difficulty**: 5/10

### Cutting-Edge Research Methods (2024-2026)

#### 20. Prediction-Powered Inference (PPI)
- **Paper**: Angelopoulos et al. (2023) "Prediction-Powered Inference", ICLR 2025 extensions
- **Used by**: Research frontier (Meta, Stanford)
- **What**: Uses ML predictions to augment small labeled datasets for valid inference.
  Relevant for A/B tests where only a subset of users have outcome data (delayed conversions).
- **Algorithm**: Combine ML-imputed outcomes with a small labeled set for debiased
  estimation with valid confidence intervals.
- **Module**: `splita.variance.ppi`
- **Class**: `PredictionPoweredInference`
- **Difficulty**: 7/10

#### 21. Optimal Proxy Metrics (Powerful Metrics)
- **Paper**: Jeunen (2024) "Powerful A/B-Testing Metrics and Where to Find Them"
- **Used by**: ShareChat, research frontier
- **What**: Learns "optimal" short-term proxy metrics from historical experiments
  that maximize statistical power for detecting effects on the north star metric.
- **Algorithm**: Optimize a weighted combination of candidate metrics to maximize
  correlation with the north star, then use as an OEC.
- **Module**: Extend `splita.core.oec`
- **Difficulty**: 6/10

#### 22. Bayesian Optimization for Long-term Outcomes
- **Paper**: Meta (2025) "Experimenting, Fast and Slow"
- **Used by**: Meta
- **What**: Combines short-running and long-running experiments using Bayesian
  optimization to find optimal treatments while targeting long-term outcomes.
  Reduces experimentation wall time by 60%+.
- **Algorithm**: Multi-fidelity Bayesian optimization with Gaussian processes,
  short experiments as cheap evaluations, long experiments as expensive ones.
- **Module**: `splita.design.bayesian_optimization`
- **Class**: `BayesianExperimentOptimizer`
- **Difficulty**: 8/10

#### 23. Automated Randomization Validation
- **Paper**: Microsoft (2022) "Ensure A/B Test Quality at Scale"
- **Used by**: Microsoft
- **What**: Automated pre-test and during-test quality checks: population stability
  index (PSI), sample ratio mismatch via sequential analysis, covariate balance checks.
- **Algorithm**: PSI for distribution drift, sequential SRM with alpha-spending,
  standardized mean difference for covariate balance.
- **Module**: Extend `splita.diagnostics`
- **Class**: `RandomizationValidator`
- **Difficulty**: 4/10

#### 24. Experimentation Accelerator (AI-Powered)
- **Paper**: arxiv 2602.13852 (2026)
- **Used by**: Research frontier
- **What**: Uses content embeddings and historical A/B results to prioritize
  which variants to test, explain why winners win, and suggest new variants.
- **Algorithm**: Content-aware ranking with semantic similarity, transfer learning
  from historical experiments.
- **Module**: Out of scope for v0.5 (requires LLM/embedding infrastructure)
- **Difficulty**: 9/10

#### 25. Combining RCTs and Observational Data
- **Paper**: Rosenman et al. (2025 review), Degtiar & Rose (2023)
- **Used by**: Research frontier
- **What**: Combines experimental (high internal validity) and observational data
  (high external validity) for better generalizability. Transports treatment effects
  from experiment population to target population.
- **Algorithm**: Inverse probability weighting for population transport,
  doubly robust estimators, calibration weighting.
- **Module**: `splita.causal.transportability`
- **Class**: `EffectTransport`
- **Difficulty**: 7/10

### Advanced Causal Estimators

#### 26. Doubly Robust / AIPW Estimator
- **Paper**: Robins et al. (1994), Bang & Robins (2005); Tutorial: arxiv 2406.00853 (2024)
- **Used by**: Microsoft (EconML), Uber (CausalML), Stanford (grf)
- **What**: Combines outcome modeling AND propensity weighting. Unbiased if EITHER
  model is correctly specified. The gold standard for observational causal inference.
- **Algorithm**: `ATE = 1/n * sum[ (T_i/e(X_i) - (1-T_i)/(1-e(X_i))) * Y_i -
  ((T_i - e(X_i)) / (e(X_i)*(1-e(X_i)))) * (mu_1(X_i) - mu_0(X_i)) ]`
- **Module**: `splita.causal.doubly_robust`
- **Class**: `DoublyRobustEstimator`
- **Difficulty**: 6/10

#### 27. TMLE (Targeted Maximum Likelihood Estimation)
- **Paper**: van der Laan & Rubin (2006), van der Laan & Rose (2011)
- **Used by**: Research frontier, epidemiology
- **What**: Two-step procedure: (1) fit outcome + treatment models, (2) targeted bias
  correction step. Compatible with machine learning. Asymptotically efficient.
- **Algorithm**: Initial estimate → clever covariate → fluctuation → update.
- **Module**: `splita.causal.tmle`
- **Class**: `TMLE`
- **Difficulty**: 7/10

### Adaptive Experiment Design

#### 28. Sample Size Re-estimation
- **Paper**: Mehta & Pocock (2011), rpact R package
- **Used by**: Clinical trials, mature experimentation platforms
- **What**: Mid-experiment sample size adjustment based on interim results. If the
  effect is smaller than expected, increase n to maintain power. Preserves Type I error.
- **Algorithm**: Conditional power at interim → re-estimate required n → adjust.
  Uses Chen-DeMets-Lan method for Type I error control.
- **Module**: Extend `splita.sequential.group_sequential`
- **Class**: `SampleSizeReestimation`
- **Difficulty**: 6/10

#### 29. Adaptive Enrichment Design
- **Paper**: Simon & Simon (2013), Bayesian enrichment (arxiv 2603.09919, 2025)
- **Used by**: Clinical trials, precision targeting
- **What**: Mid-experiment population selection. If treatment works better in a
  subgroup, enrich the trial by recruiting more from that subgroup. Stops recruiting
  from subgroups where treatment doesn't work.
- **Algorithm**: Stage-wise enrichment rules via conditional power or Bayesian
  posterior probability of benefit.
- **Module**: `splita.design.adaptive_enrichment`
- **Class**: `AdaptiveEnrichment`
- **Difficulty**: 8/10

#### 30. Response-Adaptive Randomization
- **Paper**: Hu & Rosenberger (2006), Thompson (1933)
- **Used by**: Clinical trials, ethical experimentation
- **What**: Update randomization probabilities based on accumulating data.
  More patients assigned to the better-performing arm. Bridges bandits and
  clinical trial design.
- **Algorithm**: Bayesian response-adaptive randomization using posterior
  probability of superiority. Generalizes Thompson Sampling to trial design.
- **Module**: `splita.design.response_adaptive`
- **Class**: `ResponseAdaptiveRandomization`
- **Difficulty**: 5/10

### Funnel & Journey Analysis

#### 31. Funnel Experiment Analysis
- **Paper**: Industry standard (Optimizely, VWO, Statsig)
- **Used by**: All e-commerce platforms
- **What**: Analyze treatment effects at each step of a conversion funnel
  (landing → product → cart → checkout → purchase). Identifies WHERE the
  treatment helps or hurts.
- **Algorithm**: Per-step conversion rate test + conditional conversion analysis +
  funnel visualization data.
- **Module**: `splita.core.funnel`
- **Class**: `FunnelExperiment`
- **Difficulty**: 4/10

### Specialized Variance Reduction

#### 32. Trimmed Mean Estimator
- **Paper**: Tukey (1960), recent: "Can we do better than the trimmed mean?" (2024)
- **Used by**: Various platforms internally
- **What**: Alternative to winsorization. Removes extreme values and computes mean
  on the remaining data. More statistically efficient than winsorized mean for
  certain distributions.
- **Algorithm**: Sort, trim alpha/2 from each end, compute mean + SE on trimmed data.
  Correct SE accounts for trimming: Staudte & Sheather (1990).
- **Module**: `splita.variance.trimmed_mean`
- **Class**: `TrimmedMeanEstimator`
- **Difficulty**: 3/10

### Survival / Time-to-Event Analysis

#### 33. SurvivalExperiment
- **Paper**: Cox (1972) "Regression models and life tables", Kaplan & Meier (1958)
- **Used by**: Subscription products, churn analysis
- **What**: When the outcome is TIME to an event (time to churn, time to purchase,
  time to upgrade). Standard A/B tests ignore censoring (users who haven't converted
  yet). Survival analysis handles this correctly.
- **Algorithm**: Kaplan-Meier curves per group, log-rank test for significance,
  Cox proportional hazards for hazard ratio + covariates.
- **Module**: `splita.core.survival`
- **Class**: `SurvivalExperiment`
- **Difficulty**: 6/10

### Mediation Analysis

#### 34. MediationAnalysis
- **Paper**: Baron & Kenny (1986), Imai et al. (2010) "A general approach to causal
  mediation analysis"
- **Used by**: Product analytics (WHY does treatment work?)
- **What**: Decomposes treatment effect into direct effect and indirect effect through
  a mediator. E.g., new checkout flow → reduced friction (mediator) → more purchases.
  Answers "why" not just "what".
- **Algorithm**: Sequential regression approach or nonparametric mediation.
  ACME (Average Causal Mediation Effect) via `scipy.stats`.
- **Module**: `splita.causal.mediation`
- **Class**: `MediationAnalysis`
- **Difficulty**: 6/10

### Exact / Permutation Tests

#### 35. PermutationTest
- **Paper**: Fisher (1935), Ernst (2004)
- **Used by**: Small sample experiments
- **What**: Exact test that makes no distributional assumptions. Permutes treatment
  labels and computes test statistic for every permutation. Valid for any sample size.
- **Algorithm**: Enumerate or sample permutations of treatment assignment, compute
  test statistic for each, p-value = proportion of permutations as extreme as observed.
- **Module**: `splita.core.permutation`
- **Class**: `PermutationTest`
- **Difficulty**: 3/10

### Hierarchical / Mixed-Effects Experiments

#### 36. MixedEffectsExperiment
- **Paper**: Statsig (2025), Bates et al. (lme4)
- **Used by**: Experiments with repeated measures per user, cross-device, multi-session
- **What**: When users have multiple observations (sessions, purchases), standard tests
  overstate significance by treating each observation as independent. Mixed-effects
  models correctly account for within-user correlation.
- **Algorithm**: Random intercept per user, fixed effect for treatment.
  Equivalent to cluster-robust SE but more efficient.
- **Module**: `splita.core.mixed_effects`
- **Class**: `MixedEffectsExperiment`
- **Difficulty**: 6/10

### Multi-Factor Experiment Design

#### 37. FractionalFactorialDesign
- **Paper**: Box & Hunter (1961), MOST framework (Collins 2018)
- **Used by**: Testing multiple factors simultaneously (feature flags, UI elements)
- **What**: When you want to test N factors (e.g., button color, header text, layout)
  with 2^N cells being infeasible. Fractional factorial designs test a strategically
  chosen subset that can still estimate main effects and key interactions.
- **Algorithm**: Generate design matrix, assign users, analyze main effects and
  interactions via ANOVA or regression.
- **Module**: `splita.design.factorial`
- **Class**: `FractionalFactorialDesign`
- **Difficulty**: 5/10

### Continuous Treatment Effects

#### 38. ContinuousTreatmentEffect
- **Paper**: Hirano & Imbens (2004) "The propensity score with continuous treatments"
- **Used by**: Pricing, dosage, ad frequency experiments
- **What**: When treatment is not binary but continuous (e.g., discount percentage,
  ad impressions, notification frequency). Estimates the dose-response curve.
- **Algorithm**: Generalized propensity score, kernel-weighted regression,
  or B-spline regression of outcome on treatment dose.
- **Module**: `splita.causal.continuous_treatment`
- **Class**: `ContinuousTreatmentEffect`
- **Difficulty**: 6/10

### Research Integrity

#### 39. PHackingDetector
- **Paper**: Simonsohn et al. (2014) "P-curve", Head et al. (2015)
- **Used by**: Research teams, experiment review boards
- **What**: Detects potential p-hacking in a collection of experiments.
  Analyzes the distribution of p-values — if there's a suspicious spike
  just below 0.05, it suggests selective reporting.
- **Algorithm**: P-curve analysis (distribution of significant p-values should
  be right-skewed under true effects), binomial test for p-value bunching,
  Caliper test around 0.05.
- **Module**: `splita.diagnostics.phacking`
- **Class**: `PHackingDetector`
- **Difficulty**: 4/10

### Offline / Counterfactual Evaluation

#### 40. OfflineEvaluator
- **Paper**: Li et al. (2011) "Unbiased offline evaluation of contextual-bandit-based
  news article recommendation", Gilotte et al. (2018)
- **Used by**: Recommendation systems, ad platforms
- **What**: Evaluate a new policy (ranking, recommendation) using historical logged data
  without deploying it. Uses inverse propensity scoring to debias logged data.
- **Algorithm**: IPS estimator: `V(pi) = 1/n * sum(r_i * pi(a_i|x_i) / mu(a_i|x_i))`
  where mu is the logging policy. Doubly robust version for lower variance.
- **Module**: `splita.bandits.offline_evaluation`
- **Class**: `OfflineEvaluator`
- **Difficulty**: 7/10

### Novel Sequential Methods

#### 41. YEAST Sequential Test
- **Paper**: Kurennoy (Meta) — "YEAST: Yet Another Sequential Test" (arxiv 2406.16523, 2024)
- **Used by**: Meta, major e-commerce platform
- **What**: Novel sequential test that outperforms mSPRT. Uses constant or staircase
  significance boundaries. No tuning parameters. Works for any metric type including
  real-valued financial metrics (revenue). Based on generalized Levy's inequalities.
- **Algorithm**: Constructs confidence sequence using inverted Levy bounds. Supports
  unlimited interim checks. All inputs estimable from pre-experiment data.
- **Module**: `splita.sequential.yeast`
- **Class**: `YEAST`
- **Difficulty**: 5/10

### Two-Sided Marketplace Methods

#### 42. MarketplaceExperiment
- **Paper**: Bajari et al. (2023) "Experimental Design in Two-Sided Platforms" (Management Science);
  Johari et al. (2022) "Interference, Bias, and Variance" (WWW)
- **Used by**: Uber, Lyft, Airbnb, DoorDash
- **What**: Framework for choosing buyer-side vs seller-side randomization in two-sided
  marketplaces. Analyzes bias-variance tradeoff. Recommends optimal randomization side
  based on market balance (supply-constrained vs demand-constrained).
- **Algorithm**: Market model → compute expected bias under each randomization design →
  select bias-minimizing design → adjust variance for interference.
- **Module**: `splita.causal.marketplace`
- **Class**: `MarketplaceExperiment`
- **Difficulty**: 7/10

#### 43. BudgetSplitDesign
- **Paper**: Liu et al. (LinkedIn) — "Trustworthy Online Marketplace Experimentation
  with Budget-split Design" (KDD 2021)
- **Used by**: LinkedIn
- **What**: Creates two independent sub-marketplaces (treatment and control) to eliminate
  cannibalization bias. Each sub-marketplace operates with its own budget, preventing
  interference between groups. More powerful than advertiser-side randomization.
- **Algorithm**: Split advertiser budgets proportionally → create isolated sub-campaigns →
  compare marketplace-level outcomes. Delta method for SE.
- **Module**: `splita.design.budget_split`
- **Class**: `BudgetSplitDesign`
- **Difficulty**: 6/10

#### 44. BipartiteExperiment
- **Paper**: Harshaw et al. (2023); Vinted (2024) "Measuring Sell Side Outcomes
  in Buy Side Marketplace Experiments"
- **Used by**: Vinted, marketplaces
- **What**: When randomization units (buyers) differ from outcome units (sellers).
  Constructs in-experiment bipartite graph to measure cross-side effects.
  Novel direction at intersection of bipartite experiments and mediation analysis.
- **Algorithm**: Build bipartite graph from transaction data → Horvitz-Thompson
  estimator with exposure mapping → cross-side effect estimation.
- **Module**: `splita.causal.bipartite`
- **Class**: `BipartiteExperiment`
- **Difficulty**: 7/10

---

## Complete Build Backlog (all versions)

| # | Method | Version | Difficulty | Source |
|---|--------|---------|-----------|--------|
| 1 | GuardrailMonitor | v0.4 | 3/10 | Microsoft/all |
| 2 | FlickerDetector | v0.4 | 3/10 | All platforms |
| 3 | TrimmedMeanEstimator | v0.4 | 3/10 | Tukey 1960 |
| 4 | OECBuilder | v0.4 | 4/10 | Microsoft/LinkedIn |
| 5 | Dilution Analysis | v0.4 | 4/10 | Microsoft/Uber |
| 6 | PostStratification | v0.4 | 4/10 | Alibaba |
| 7 | ClusterBootstrap | v0.4 | 4/10 | Uber |
| 8 | RandomizationValidator | v0.4 | 4/10 | Microsoft 2022 |
| 9 | FunnelExperiment | v0.4 | 4/10 | Industry standard |
| 10 | InterleavingExperiment | v0.4 | 5/10 | Airbnb/Netflix |
| 11 | CarryoverDetector | v0.4 | 5/10 | Uber/LinkedIn |
| 12 | RobustMeanEstimator | v0.4 | 5/10 | Huber 1964 |
| 13 | InstrumentalVariables | v0.4 | 5/10 | Angrist 2009 |
| 14 | PropensityScoreMatching | v0.4 | 5/10 | Rosenbaum 1983 |
| 15 | ResponseAdaptiveRandomization | v0.4 | 5/10 | Hu & Rosenberger 2006 |
| 16 | InExperimentVR (INEX) | v0.4 | 6/10 | Microsoft KDD'23 |
| 17 | MetricDecomposition | v0.4 | 6/10 | Microsoft KDD'24 |
| 18 | ObjectiveBayesianExperiment | v0.4 | 6/10 | Microsoft/Bing |
| 19 | RiskAwareDecision | v0.4 | 6/10 | Spotify 2024 |
| 20 | OptimalProxyMetrics | v0.4 | 6/10 | Jeunen 2024 |
| 21 | DoublyRobustEstimator | v0.4 | 6/10 | Robins 1994 |
| 22 | SampleSizeReestimation | v0.4 | 6/10 | Mehta & Pocock 2011 |
| 23 | RegressionDiscontinuity | v0.5 | 6/10 | Imbens 2008 |
| 24 | NonstationaryAdjustment | v0.5 | 7/10 | Microsoft 2024 |
| 25 | GeoExperiment | v0.5 | 7/10 | Google |
| 26 | PredictionPoweredInference | v0.5 | 7/10 | Angelopoulos 2023 |
| 27 | EffectTransport | v0.5 | 7/10 | Rosenman 2025 |
| 28 | TMLE | v0.5 | 7/10 | van der Laan 2006 |
| 29 | DynamicCausalEffect | v0.5 | 8/10 | Microsoft JASA'22 |
| 30 | BayesianExperimentOptimizer | v0.5 | 8/10 | Meta 2025 |
| 31 | AdaptiveEnrichment | v0.5 | 8/10 | Simon 2013 |
| 32 | SurvivalExperiment | v0.5 | 6/10 | Cox 1972 |
| 33 | MediationAnalysis | v0.5 | 6/10 | Baron & Kenny 1986 |
| 34 | PermutationTest | v0.4 | 3/10 | Fisher 1935 |
| 35 | MixedEffectsExperiment | v0.5 | 6/10 | Statsig 2025 |
| 36 | FractionalFactorialDesign | v0.5 | 5/10 | Box 1961 |
| 37 | ContinuousTreatmentEffect | v0.5 | 6/10 | Hirano & Imbens 2004 |
| 38 | PHackingDetector | v0.4 | 4/10 | Research integrity |
| 39 | OfflineEvaluator | v0.5 | 7/10 | Li et al. 2011 |
| 40 | OfflineEvaluator | v0.5 | 7/10 | Li et al. 2011 |
| 41 | YEASTSequentialTest | v0.4 | 5/10 | Meta (arxiv 2024) |
| 42 | MarketplaceExperiment | v0.5 | 7/10 | Stanford/Airbnb (WWW 2022) |
| 43 | BudgetSplitDesign | v0.5 | 6/10 | LinkedIn (KDD 2021) |
| 44 | BipartiteExperiment | v0.5 | 7/10 | Vinted (2024) |
| 45 | ExperimentationAccelerator | v0.6+ | 9/10 | arxiv 2026 |

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
