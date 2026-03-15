# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-15

### Added

- Core experimentation engine (`Experiment`, `BayesianExperiment`, `QuantileExperiment`).
- Sequential testing methods (`mSPRT`, `EValue`, `EProcess`, `GroupSequential`, `ConfidenceSequence`, `YEAST`).
- Variance reduction techniques (`CUPED`, `CUPAC`, `DoubleML`, `PostStratification`, `RegressionAdjustment`).
- Causal inference suite (`DifferenceInDifferences`, `SyntheticControl`, `CausalForest`, `TMLE`, `DoublyRobustEstimator`).
- Experiment diagnostics (`AATest`, `SRMCheck`, `FlickerDetector`, `NoveltyCurve`, `PHackingDetector`).
- Multi-armed bandit algorithms (`ThompsonSampler`, `LinUCB`, `LinTS`).
- Experiment design helpers (`FractionalFactorialDesign`, `PairwiseDesign`, `BudgetSplitDesign`).
- Governance layer (`ExperimentRegistry`, `GuardrailMonitor`, `ConflictDetector`).
- Structured exception hierarchy (`errors.py`).
- Frozen dataclass result types for all public APIs.
- Comprehensive validation utilities (`_validation.py`).
