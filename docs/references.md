# splita — Methods Reference

Every method in splita is either an **original implementation** from a published paper,
a **wrapper** around scipy/sklearn with added validation and A/B-testing-specific logic,
or a **hybrid** that implements core algorithms from scratch but uses scipy/sklearn for
numerical primitives (solvers, distributions).

Legend:
- **Original**: Algorithm implemented from the paper's equations. No delegation to existing statistical libraries for the core logic.
- **Wrapper**: Delegates the statistical computation to scipy or sklearn. splita adds validation, auto-detection, error messages, and result formatting.
- **Hybrid**: Core algorithm from paper + scipy/sklearn for numerical primitives (e.g., `norm.ppf`, `linalg.solve`).

---

## Core Analysis (`splita.core`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `Experiment` (z-test) | Hybrid | Pooled SE test statistic, unpooled SE for CI | Newcombe, R.G. (1998). "Two-sided confidence intervals for the single proportion." *Statistics in Medicine*, 17(8), 857-872. |
| `Experiment` (t-test) | Wrapper | Welch's t-test via `scipy.stats.ttest_ind` | Welch, B.L. (1947). "The generalization of Student's problem when several different population variances are involved." *Biometrika*, 34(1-2), 28-35. |
| `Experiment` (Mann-Whitney) | Hybrid | P-value via `scipy.stats.mannwhitneyu`, Hodges-Lehmann estimator + Moses CI implemented from scratch | Hodges, J.L. & Lehmann, E.L. (1963). "Estimates of location based on rank tests." *Annals of Mathematical Statistics*, 34(2), 598-611. Moses, L.E. (1965). "Confidence limits from rank tests." *Technometrics*, 7(2), 257-260. |
| `Experiment` (chi-square) | Wrapper | `scipy.stats.chi2_contingency` | Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables." *Philosophical Magazine*, 50(302), 157-175. |
| `Experiment` (delta method) | Original | Linearized ratio metric, Welch t-test on linearized values | Deng, A., Knoblich, U., & Lu, J. (2018). "Applying the Delta Method in Metric Analytics." *KDD '18*. |
| `Experiment` (bootstrap) | Original | Vectorized resampling, shifted bootstrap p-value, percentile CI | Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. |
| `BayesianExperiment` | Original | Beta-Binomial and Normal-Inverse-Gamma conjugate posteriors, MC inference | Berry, D.A. (2006). "Bayesian clinical trials." *Nature Reviews Drug Discovery*, 5(1), 27-36. |
| `QuantileExperiment` | Original | Bootstrap inference for quantile differences at arbitrary quantiles | Efron, B. (1979). "Bootstrap methods: Another look at the jackknife." *Annals of Statistics*, 7(1), 1-26. |
| `SampleSize` (proportion) | Hybrid | Farrington-Manning formula with pooled/unpooled SE split | Farrington, C.P. & Manning, G. (1990). "Test statistics and sample size formulae for comparative binomial trials." *Statistics in Medicine*, 9(12), 1447-1454. |
| `SampleSize` (mean) | Hybrid | Two-sample t-test power formula | Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed. Lawrence Erlbaum. |
| `SampleSize` (ratio) | Original | Delta method variance for ratio metrics | Deng, A., Knoblich, U., & Lu, J. (2018). "Applying the Delta Method in Metric Analytics." *KDD '18*. |
| `SampleSize` (MDE inverse) | Hybrid | Numerical inversion via `scipy.optimize.brentq` | Brent, R.P. (1973). *Algorithms for Minimization Without Derivatives*. Prentice-Hall. |
| `SRMCheck` | Wrapper | Chi-square goodness-of-fit via `scipy.stats.chi2.sf` | Fabijan, A. et al. (2019). "Diagnosing Sample Ratio Mismatch in Online Controlled Experiments." *WWW '19*. |
| `MultipleCorrection` (BH) | Original | Step-up procedure with reverse monotonicity enforcement | Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery rate." *JRSS-B*, 57(1), 289-300. |
| `MultipleCorrection` (Bonferroni) | Original | p * n, capped at 1 | Bonferroni, C.E. (1936). "Teoria statistica delle classi e calcolo delle probabilita." *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*. |
| `MultipleCorrection` (Holm) | Original | Step-down procedure with forward monotonicity enforcement | Holm, S. (1979). "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70. |
| `MultipleCorrection` (BY) | Original | BH with harmonic number correction for dependent tests | Benjamini, Y. & Yekutieli, D. (2001). "The control of the false discovery rate in multiple testing under dependency." *Annals of Statistics*, 29(4), 1165-1188. |
| `PowerSimulation` | Wrapper | Monte Carlo simulation using `Experiment` internally | — (standard simulation methodology) |
| `HTEEstimator` (T-learner) | Wrapper | Two separate sklearn models, CATE = E[Y|X,T=1] - E[Y|X,T=0] | Kunzel, S.R. et al. (2019). "Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning." *PNAS*, 116(10), 4156-4165. |
| `HTEEstimator` (S-learner) | Wrapper | Single sklearn model with treatment indicator as feature | Kunzel, S.R. et al. (2019). Same as above. |
| `TriggeredExperiment` | Wrapper | ITT and per-protocol analysis via `Experiment` | Hernan, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall. |
| `InteractionTest` | Hybrid | Per-segment experiments + Cochran's Q heterogeneity test | Cochran, W.G. (1954). "The combination of estimates from different experiments." *Biometrics*, 10(1), 101-129. |
| `MultiObjectiveExperiment` | Wrapper | Runs `Experiment` per metric + `MultipleCorrection` + Pareto analysis | — (composite of existing methods) |
| `StratifiedExperiment` | Original | Neyman-style stratified difference-in-means with weighted variance | Neyman, J. (1923/1990). "On the application of probability theory to agricultural experiments." *Statistical Science*, 5(4), 465-472. Miratrix, L.W. et al. (2013). "Adjusting treatment effect estimates by post-stratification in randomized experiments." *JRSS-B*, 75(2), 369-396. |
| `CausalForest` | Hybrid | T-learner with `sklearn.RandomForestRegressor` + honest splitting + jackknife CI | Athey, S., Tibshirani, J., & Wager, S. (2019). "Generalized Random Forests." *Annals of Statistics*, 47(2), 1148-1178. Wager, S. & Athey, S. (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *JASA*, 113(523), 1228-1242. |

## Variance Reduction (`splita.variance`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `CUPED` | Original | Y_adj = Y - theta*(X - mean(X)), theta = Cov(Y,X)/Var(X) | Deng, A. et al. (2013). "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data." *WSDM '13*. |
| `CUPAC` | Hybrid | Cross-validated ML predictions as CUPED covariate. sklearn for models. | Tang, D. et al. (2020). "An empirical evaluation of CUPAC." DoorDash Engineering Blog. Guo, Y. et al. (2021). "Machine Learning for Variance Reduction in Online Experiments." *NeurIPS '21*. |
| `OutlierHandler` (winsorize/trim) | Original | Percentile-based capping on pooled data | Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley. (IQR rule) |
| `OutlierHandler` (clustering) | Wrapper | DBSCAN via `sklearn.cluster.DBSCAN` for outlier detection | Ester, M. et al. (1996). "A density-based algorithm for discovering clusters." *KDD '96*. |
| `MultivariateCUPED` | Original | theta = Cov(Y,X) @ Var(X)^{-1}, multivariate extension | Deng, A. & Shi, X. (2016). "Optimal Variance Reduction for Online Controlled Experiments." Microsoft Technical Report. Poyarkov, A. et al. (2016). "Boosted Decision Tree Regression Adjustment for Variance Reduction." *KDD '16*. |
| `RegressionAdjustment` | Original | Fully-interacted OLS with HC2 robust standard errors | Lin, W. (2013). "Agnostic notes on regression adjustments to experimental data." *Annals of Applied Statistics*, 7(1), 295-318. |
| `AdaptiveWinsorizer` | Original | Grid-search optimal capping thresholds to minimize effect variance | Gupta, S. et al. (2019). "Top Challenges from the first Practical Online Controlled Experiments Summit." *KDD '19*. |
| `DoubleML` | Hybrid | Cross-fitted outcome + propensity models, influence-function SE. sklearn for models. | Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." *Econometrics Journal*, 21(1), C1-C68. |

## Sequential Testing (`splita.sequential`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `mSPRT` | Original | Mixture likelihood ratio, always-valid p-values, streaming API | Johari, R., Pekelis, L., & Walsh, D.J. (2017/2022). "Always Valid Inference: Continuous Monitoring of A/B Tests." *Operations Research*, 70(3), 1806-1821. (arXiv:1512.04922) |
| `GroupSequential` | Original | Conditional error spending boundaries (Lan-DeMets approach) | O'Brien, P.C. & Fleming, T.R. (1979). "A multiple testing procedure for clinical trials." *Biometrics*, 35(3), 549-556. Lan, K.K.G. & DeMets, D.L. (1983). "Discrete sequential boundaries for clinical trials." *Biometrika*, 70(3), 659-663. |
| `EValue` | Original | E-value = mixture likelihood ratio, always-valid testing | Vovk, V. & Wang, R. (2021). "E-values: Calibration, combination, and applications." *Annals of Statistics*, 49(3), 1736-1754. Grunwald, P., de Heide, R., & Koolen, W. (2020). "Safe Testing." arXiv:1906.07801. |
| `ConfidenceSequence` | Original | Time-uniform confidence sequences, tighter than mSPRT CIs | Howard, S.R. et al. (2021). "Time-uniform, nonparametric, nonasymptotic confidence sequences." *Annals of Statistics*, 49(2), 1055-1080. |
| `EProcess` | Original | Multiplicative e-value accumulation, GRAPA and universal methods | Grunwald, P., de Heide, R., & Koolen, W. (2020). "Safe Testing." arXiv:1906.07801. Ramdas, A. et al. (2023). "Game-theoretic Statistics and Safe Anytime-valid Inference." *Statistical Science*. |

## Bandits (`splita.bandits`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `ThompsonSampler` | Original | Beta-Binomial, Normal-Inverse-Gamma, Gamma-Poisson conjugate posteriors | Russo, D. et al. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*, 11(1), 1-96. Thompson, W.R. (1933). "On the likelihood that one unknown probability exceeds another." *Biometrika*, 25(3-4), 285-294. |
| `LinTS` | Original | Bayesian linear regression posterior with Cholesky sampling | Agrawal, S. & Goyal, N. (2013). "Thompson Sampling for Contextual Bandits with Linear Payoffs." *ICML '13*. |
| `LinUCB` | Original | Upper confidence bound on linear reward model | Li, L. et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW '10*. |
| `BayesianStopping` | Original | Standalone stopping rule evaluator (expected loss, prob best, precision) | — (standard Bayesian decision theory) |

## Causal Inference (`splita.causal`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `DifferenceInDifferences` | Hybrid | Classic two-period DiD with delta-method SE + parallel trends check | Card, D. & Krueger, A.B. (1994). "Minimum Wages and Employment." *American Economic Review*, 84(4), 772-793. Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. |
| `SyntheticControl` | Hybrid | Constrained optimization (SLSQP) for donor weights, pre/post comparison | Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies." *JASA*, 105(490), 493-505. |
| `ClusterExperiment` | Hybrid | Cluster-robust inference via cluster-mean collapse + Welch t-test + ICC | Cameron, A.C. & Miller, D.L. (2015). "A practitioner's guide to cluster-robust inference." *Journal of Human Resources*, 50(2), 317-372. |
| `SwitchbackExperiment` | Hybrid | Period-level averaging + t-test on period means | Bojinov, I. & Shephard, N. (2019). "Time series experiments and causal estimands." *JASA*, 114(528), 1477-1491. |
| `SurrogateEstimator` | Wrapper | sklearn model mapping short-term → long-term outcome | Athey, S. et al. (2019). "The Surrogate Index." NBER Working Paper 26463. |
| `SurrogateIndex` | Hybrid | Cross-fitted multi-surrogate index with delta-method CI | Athey, S., Chetty, R., Imbens, G.W., & Kang, H. (2019). "The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term Treatment Effects." NBER Working Paper 26463. |
| `InterferenceExperiment` | Original | Horvitz-Thompson at cluster level with ICC-based design effect | Basse, G.W. & Feller, A. (2018). "Analyzing Two-Stage Experiments in the Presence of Interference." *JASA*, 113(521), 41-55. Hudgens, M.G. & Halloran, M.E. (2008). "Toward Causal Inference With Interference." *JASA*, 103(482), 832-842. |

## Diagnostics (`splita.diagnostics`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `NoveltyCurve` | Original | Rolling-window effect analysis with trend detection | — (standard diagnostic methodology, used at Booking.com, Microsoft) |
| `AATest` | Wrapper | Random-split simulations using `Experiment` to validate FP rate | — (standard pre-experiment validation, described in Kohavi et al. 2020) |
| `EffectTimeSeries` | Hybrid | Cumulative experiment at each timestamp | — (standard diagnostic, described in Kohavi et al. 2020) |
| `MetricSensitivity` | Hybrid | Monte Carlo power estimation from historical variance | — (standard pre-experiment planning) |
| `VarianceEstimator` | Original | Distributional analysis with skewness/kurtosis diagnostics | — (standard descriptive statistics with A/B-specific recommendations) |
| `NonStationaryDetector` | Original | CUSUM-like change-point detection on effect time series | Page, E.S. (1954). "Continuous inspection schemes." *Biometrika*, 41(1-2), 100-115. (CUSUM) |

## Experiment Design (`splita.design`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `PairwiseDesign` | Original | Mahalanobis distance greedy matching for balanced assignment | Greevy, R. et al. (2004). "Optimal multivariate matching before randomization." *Biostatistics*, 5(2), 263-275. Mahalanobis, P.C. (1936). "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India*, 2, 49-55. |

## Experiment Governance (`splita.governance`)

| Class | Type | What it does | Reference |
|-------|------|-------------|-----------|
| `ExperimentRegistry` | Original | In-memory experiment tracking with date filtering | — (operational tooling, no academic reference) |
| `ConflictDetector` | Original | Overlapping experiment detection (traffic, metric, segment conflicts) | — (operational tooling, described in Kohavi, Tang & Xu 2020) |

---

## General References

- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed. Lawrence Erlbaum.
- Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
