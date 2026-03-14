# splita v0.3.0 — Research-Grade Methods PRD

**Date**: 2026-03-14
**Status**: Planned

Methods from recent research papers and R packages that are either not implemented
in Python at all, or only available in fragmented/academic code.

---

## Methods Ranked by Difficulty

### Tier 1: Easy (1-2 hours each, well-understood math)

#### 1. Lin's Regression Adjustment
- **Paper**: Lin (2013) "Agnostic notes on regression adjustments to experimental data"
- **What**: OLS regression adjustment post-stratification. Provably at least as efficient as CUPED, often better with multiple covariates.
- **Formula**: Regress Y on treatment indicator + covariates + treatment*covariate interactions, read off the treatment coefficient. Uses HC2 robust standard errors.
- **Why no one has it**: Everyone uses CUPED instead, but Lin's method is strictly better when you have multiple covariates.
- **Module**: `splita.variance.regression_adjustment`
- **Class**: `RegressionAdjustment`
- **Dependencies**: numpy, scipy (OLS via normal equations)
- **Difficulty**: 2/10

#### 2. Stratified Randomization Analyzer
- **Paper**: Zhao et al. (2023) "Rerandomization and regression adjustment"
- **What**: When you pre-stratify (block) your experiment, standard SEs are conservative. This computes the correct (smaller) SEs.
- **Formula**: Stratified difference-in-means with Neyman-style variance estimator within each stratum, then combine.
- **Module**: `splita.core.stratified`
- **Class**: `StratifiedExperiment`
- **Dependencies**: numpy, scipy
- **Difficulty**: 3/10

#### 3. Variance-Weighted Estimator
- **Paper**: Deng & Shi (2016) "Optimal variance reduction for online controlled experiments"
- **What**: When you have multiple pre-experiment covariates, jointly optimizes the CUPED theta vector instead of using a single covariate.
- **Formula**: theta = Cov(Y, X) @ Var(X)^{-1} (multivariate extension of CUPED's scalar theta)
- **Module**: `splita.variance.multivariate_cuped`
- **Class**: `MultivariateCUPED`
- **Dependencies**: numpy (matrix operations)
- **Difficulty**: 3/10

#### 4. Adaptive Metric Winsorization
- **Paper**: Microsoft ExP platform (Gupta et al. 2019) "Top Challenges from the first Practical Online Controlled Experiments Summit"
- **What**: Instead of fixed percentile thresholds, learn optimal capping thresholds that minimize MSE of the treatment effect estimator.
- **Formula**: Optimize `threshold = argmin Var(Y_capped_trt - Y_capped_ctrl)` via grid search or golden section.
- **Module**: `splita.variance.adaptive_winsorization`
- **Class**: `AdaptiveWinsorizer`
- **Dependencies**: numpy
- **Difficulty**: 3/10

### Tier 2: Medium (3-5 hours each, requires careful implementation)

#### 5. Confidence Sequences
- **Paper**: Howard, Ramdas, McAuliffe, Sekhon (2021) "Time-uniform, nonparametric, nonasymptotic confidence sequences"
- **What**: Always-valid CIs that are tighter than mSPRT. Work for any distribution, not just normal. The current frontier of sequential testing.
- **Key idea**: Uses sub-exponential or sub-Gaussian tail bounds with a boundary function that grows as O(sqrt(n * log(log(n)))).
- **Formula**: CI at time t: `x_bar_t ± sqrt(2 * sigma^2 * (log(log(2*t)) + 0.72*log(5.2/alpha)) / t)` (normal mixture variant)
- **Module**: `splita.sequential.confidence_sequence`
- **Class**: `ConfidenceSequence`
- **Dependencies**: numpy, scipy
- **Difficulty**: 5/10

#### 6. Double Machine Learning
- **Paper**: Chernozhukov et al. (2018) "Double/Debiased Machine Learning for Treatment and Structural Parameters"
- **What**: CUPAC on steroids. Uses cross-fitting to debias ML predictions of both outcome AND treatment propensity. Provably sqrt(n)-consistent even with slow ML learners.
- **Algorithm**: (1) Cross-fit E[Y|X], (2) Cross-fit E[T|X], (3) Compute residuals, (4) Regress Y-residual on T-residual.
- **Module**: `splita.variance.double_ml`
- **Class**: `DoubleML`
- **Dependencies**: numpy, scipy, sklearn (optional)
- **Difficulty**: 5/10

#### 7. Exact Group Sequential Boundaries
- **Paper**: Jennison & Turnbull (1999) "Group Sequential Methods with Applications to Clinical Trials"
- **What**: Replace our approximate conditional error spending with exact multivariate normal integration.
- **Algorithm**: Recursive numerical integration over the joint density of correlated z-statistics.
- **Module**: Update existing `splita.sequential.group_sequential`
- **Dependencies**: scipy (multivariate_normal, quad)
- **Difficulty**: 6/10

#### 8. Safe Testing (Full E-Process Framework)
- **Paper**: Grunwald, de Heide, Koolen (2020) "Safe Testing"
- **What**: Extends our basic EValue to full e-processes: product e-values, composite nulls, safe confidence intervals, anytime-valid testing under model misspecification.
- **Key concepts**: E-processes (multiplicative accumulation), growth-rate optimal e-values, GROW (Growth-Rate Optimal in the Worst case).
- **Module**: `splita.sequential.safe_testing`
- **Class**: `SafeTest`, `EProcess`
- **Dependencies**: numpy, scipy
- **Difficulty**: 6/10

#### 9. Multi-Objective Experimentation
- **Paper**: Letham & Bakshy (2019) "Bayesian Optimization of Function Networks"; also Daulton et al. (2022) "Multi-Objective Bayesian Optimization"
- **What**: When optimizing multiple metrics (conversion AND revenue AND engagement) with tradeoffs. Finds the Pareto frontier of treatments.
- **Algorithm**: Pareto frontier identification, expected hypervolume improvement, constrained optimization.
- **Module**: `splita.core.multi_objective`
- **Class**: `MultiObjectiveExperiment`
- **Dependencies**: numpy, scipy
- **Difficulty**: 6/10

### Tier 3: Hard (5-10 hours each, novel algorithms, limited reference implementations)

#### 10. Interference-Robust Testing
- **Paper**: Basse & Feller (2018) "Analyzing two-stage experiments in the presence of interference"
- **What**: When users interact (marketplace, social network), standard i.i.d. assumptions fail. This provides valid inference under interference.
- **Algorithm**: Two-stage randomization (randomize clusters, then users within clusters), Horvitz-Thompson estimator with network-adjusted variance.
- **Module**: `splita.causal.interference`
- **Class**: `InterferenceExperiment`
- **Dependencies**: numpy, scipy
- **Difficulty**: 7/10

#### 11. Non-Stationary Experimentation
- **Paper**: Simchi-Levi & Wang (2023) "Bypassing the monster: A faster and simpler optimal algorithm for contextual bandits under realizability"
- **What**: When user behavior shifts during the experiment (seasonality, novelty, external events). Adapts the estimator to non-stationary environments.
- **Algorithm**: Sliding-window estimators, change-point detection, time-weighted treatment effects.
- **Module**: `splita.diagnostics.nonstationarity`
- **Class**: `NonStationaryDetector`, `TimeWeightedExperiment`
- **Dependencies**: numpy, scipy
- **Difficulty**: 7/10

#### 12. Counterfactual Surrogate Index
- **Paper**: Athey, Chetty, Imbens, Kang (2019) "The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term Treatment Effects"
- **What**: Formally constructs a surrogate index from multiple short-term outcomes to predict long-term effects. More rigorous than our basic SurrogateEstimator.
- **Algorithm**: (1) Estimate E[long_term | short_term_vector] in observational data, (2) Apply to experimental short-term data, (3) Provide valid CI via delta method.
- **Module**: Update `splita.causal.surrogate` or new `splita.causal.surrogate_index`
- **Class**: `SurrogateIndex`
- **Dependencies**: numpy, scipy, sklearn (optional)
- **Difficulty**: 7/10

#### 13. Pairwise Experiment Design
- **Paper**: Bhat et al. (2020) "Near-optimal experimental design for networks"
- **What**: Optimal traffic splitting when you have covariates. Pairs similar users and assigns one to treatment, one to control. Reduces variance without CUPED.
- **Algorithm**: Mahalanobis distance matching, greedy pair assignment, rerandomization.
- **Module**: `splita.design.pairwise`
- **Class**: `PairwiseDesign`
- **Dependencies**: numpy, scipy
- **Difficulty**: 7/10

#### 14. Causal Forest HTE with Valid Inference
- **Paper**: Athey, Tibshirani, Wager (2019) "Generalized random forests"
- **What**: Our HTEEstimator uses simple T/S-learners. Causal forests provide valid confidence intervals on individual treatment effects, honest splitting, and asymptotic normality.
- **Algorithm**: Honest random forest with treatment-effect splitting criterion, jackknife variance estimation.
- **Module**: Update `splita.core.hte` or new `splita.core.causal_forest`
- **Class**: `CausalForest`
- **Dependencies**: numpy, scipy, sklearn (for tree infrastructure)
- **Difficulty**: 8/10

---

## R Packages to Port

| R Package | Key Feature | Priority | Difficulty |
|-----------|------------|----------|------------|
| **gsDesign** | Exact group sequential boundaries | High | 6/10 |
| **rpact** | Adaptive sample size reassessment | Medium | 7/10 |
| **DeclareDesign** | Declare-diagnose-redesign framework | Low | 8/10 |
| **bayesAB** | Multiple prior families, risk functions | Medium | 4/10 |
| **grf** (causal forests) | Honest causal forests with valid CI | High | 8/10 |

---

## Summary: Build Order

| # | Method | Difficulty | Est. Time | Dependencies |
|---|--------|-----------|-----------|-------------|
| 1 | Lin's Regression Adjustment | 2/10 | 1h | numpy, scipy |
| 2 | Stratified Experiment | 3/10 | 1.5h | numpy, scipy |
| 3 | Multivariate CUPED | 3/10 | 1.5h | numpy |
| 4 | Adaptive Winsorization | 3/10 | 1.5h | numpy |
| 5 | Confidence Sequences | 5/10 | 3h | numpy, scipy |
| 6 | Double ML | 5/10 | 3h | numpy, sklearn |
| 7 | Exact GS Boundaries | 6/10 | 4h | scipy |
| 8 | Safe Testing (E-Process) | 6/10 | 4h | numpy, scipy |
| 9 | Multi-Objective | 6/10 | 4h | numpy, scipy |
| 10 | Interference-Robust | 7/10 | 5h | numpy, scipy |
| 11 | Non-Stationary | 7/10 | 5h | numpy, scipy |
| 12 | Surrogate Index | 7/10 | 5h | numpy, sklearn |
| 13 | Pairwise Design | 7/10 | 5h | numpy, scipy |
| 14 | Causal Forest | 8/10 | 8h | numpy, sklearn |
