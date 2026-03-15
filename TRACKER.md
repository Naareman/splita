# splita --- Class Readiness Tracker

Every class must pass ALL 9 gates before it's considered DONE.

## Gates

| # | Gate | Description |
|---|------|-------------|
| 1 | **Tests** | Sufficient test cases covering happy path, edge cases, validation errors, different scenarios |
| 2 | **Stats Expert** | Approved by PhD statistician --- formulas verified against references |
| 3 | **Maths Expert** | Approved by mathematician --- numerical stability, convergence, correctness |
| 4 | **Tech Lead** | Approved --- code quality, API consistency, error messages, maintainability |
| 5 | **QA/QC** | Approved --- coverage 100%, edge cases, test quality |
| 6 | **Security** | Approved --- input validation, resource limits, no injection vectors |
| 7 | **E2E Scenarios** | End-to-end scenario tests exercising real-world workflows |
| 8 | **Stat Audit** | Simulation-based calibration (Type I error, CI coverage, power) |
| 9 | **Docs + Git** | Documented (docstrings + REFERENCES.md), committed to git |

Legend: Y = done, N = not done, - = not applicable

---

### Foundation (`splita._types`, `splita._validation`, `splita._utils`)

| Class/Module | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `_types` (10 dataclasses) | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `_validation` (9 functions) | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `_utils` (7 functions) | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Core (`splita.core`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `Experiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SampleSize` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SRMCheck` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MultipleCorrection` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `BayesianExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `QuantileExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `PowerSimulation` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `HTEEstimator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `TriggeredExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `InteractionTest` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MultiObjectiveExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `StratifiedExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `CausalForest` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Variance (`splita.variance`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `CUPED` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `CUPAC` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `OutlierHandler` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MultivariateCUPED` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `RegressionAdjustment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `AdaptiveWinsorizer` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `DoubleML` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Sequential (`splita.sequential`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `mSPRT` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `GroupSequential` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `EValue` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `ConfidenceSequence` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `EProcess` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Bandits (`splita.bandits`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `ThompsonSampler` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `LinTS` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `LinUCB` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `BayesianStopping` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Diagnostics (`splita.diagnostics`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `NoveltyCurve` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `AATest` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `EffectTimeSeries` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MetricSensitivity` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `VarianceEstimator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `NonStationaryDetector` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Causal (`splita.causal`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `DifferenceInDifferences` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SyntheticControl` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `ClusterExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SwitchbackExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SurrogateEstimator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SurrogateIndex` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `InterferenceExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Design (`splita.design`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `PairwiseDesign` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Governance (`splita.governance`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `ExperimentRegistry` | Y | Y | Y | Y | Y | Y | - | - | Y | DONE |
| `ConflictDetector` | Y | Y | Y | Y | Y | Y | - | - | Y | DONE |

### Industry (`splita` --- v0.4)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `GuardrailMonitor` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `FlickerDetector` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `TrimmedMeanEstimator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `OECBuilder` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `PostStratification` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `ClusterBootstrap` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `RandomizationValidator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `FunnelExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `InterleavingExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `CarryoverDetector` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `RobustMeanEstimator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `InstrumentalVariables` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `PropensityScoreMatching` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `ResponseAdaptiveRandomization` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `InExperimentVR` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MetricDecomposition` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `ObjectiveBayesianExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `RiskAwareDecision` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `OptimalProxyMetrics` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `DoublyRobustEstimator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SampleSizeReestimation` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `RegressionDiscontinuity` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `NonstationaryAdjustment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `GeoExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `PredictionPoweredInference` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `EffectTransport` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `TMLE` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `DynamicCausalEffect` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `BayesianExperimentOptimizer` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `AdaptiveEnrichment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SurvivalExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MediationAnalysis` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `PermutationTest` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MixedEffectsExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `FractionalFactorialDesign` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `ContinuousTreatmentEffect` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `PHackingDetector` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `OfflineEvaluator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `YEASTSequentialTest` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MarketplaceExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `BudgetSplitDesign` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `BipartiteExperiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `DilutionAnalysis` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `ExperimentationAccelerator` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

---

## Summary

| Status | Count |
|--------|-------|
| DONE (all 9 gates passed) | 88 |
| TODO | 0 |
| **Total** | **88** |
