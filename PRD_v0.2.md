# splita v0.2.0 — Extended Features PRD

## New Milestones

### M8: Bayesian A/B Testing
- `BayesianExperiment` — P(B>A), expected loss, credible intervals, rope
- Conjugate models: Beta-Binomial (conversion), Normal-NIG (continuous)
- Decision rules: expected loss, probability of being best, ROPE

### M9: Experiment Diagnostics
- `NoveltyCurve` — detect novelty/primacy effects via rolling window
- `AATest` — validate randomization with historical data
- `EffectTimeSeries` — track treatment effect stability over time

### M10: Quantile Treatment Effects
- `QuantileExperiment` — test differences at arbitrary quantiles (median, p90, p99)
- Bootstrap-based inference for quantile differences

### M11: Power Simulation
- `PowerSimulation` — Monte Carlo power for complex designs
- Support: CUPED-adjusted, stratified, ratio metrics, custom DGPs

### M12: Heterogeneous Treatment Effects
- `HTEEstimator` — CATE via causal forests or meta-learners (T/S/X-learner)
- `UpliftModel` — which users benefit most from treatment

### M13: Pre-experiment Validation
- `AAValidator` — run AA tests on historical data to validate metrics
- `MetricSensitivity` — estimate metric variance and sensitivity
- `VarianceEstimator` — historical variance for power calculations

### M14: Triggered + Interaction Analysis
- `TriggeredExperiment` — ITT vs per-protocol analysis
- `InteractionTest` — does effect differ across segments

### M15: Long-run Effect Estimation
- `SurrogateEstimator` — estimate long-term effects from short-term proxies

### M16: Network Effects
- `ClusterExperiment` — cluster-randomized experiments
- `SwitchbackExperiment` — time-based switchback designs

### M17: Causal Inference
- `DifferenceInDifferences` — DiD estimator
- `SyntheticControl` — synthetic control method

### M18: Experiment Governance
- `ExperimentRegistry` — track active experiments
- `ConflictDetector` — detect overlapping experiments

### M19: Complete TODO Stubs
- `EValue` — e-value sequential testing
- `LinUCB` — upper confidence bound contextual bandit
- `BayesianStopping` — Bayesian stopping rules
- `OutlierHandler(method="clustering")` — DBSCAN outlier detection
