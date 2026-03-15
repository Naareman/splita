# splita — Class Readiness Tracker

Every class must pass ALL 9 gates before it's considered DONE.

## Gates

| # | Gate | Description |
|---|------|-------------|
| 1 | **Tests** | Sufficient test cases covering happy path, edge cases, validation errors, different scenarios |
| 2 | **Stats Expert** | Approved by PhD statistician — formulas verified against references |
| 3 | **Maths Expert** | Approved by mathematician — numerical stability, convergence, correctness |
| 4 | **Tech Lead** | Approved — code quality, API consistency, error messages, maintainability |
| 5 | **QA/QC** | Approved — coverage 100%, edge cases, test quality |
| 6 | **Security** | Approved — input validation, resource limits, no injection vectors |
| 7 | **E2E Scenarios** | End-to-end scenario tests exercising real-world workflows |
| 8 | **Stat Audit** | Simulation-based calibration (Type I error, CI coverage, power) |
| 9 | **Docs + Git** | Documented (docstrings + REFERENCES.md), committed to git |

Legend: Y = done, N = not done, - = not applicable

---

## BUILT Classes (41)

### Foundation (`splita._types`, `splita._validation`, `splita._utils`)

| Class/Module | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `_types` (10 dataclasses) | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `_validation` (9 functions) | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `_utils` (7 functions) | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Core (`splita.core`) — M1-M3

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `Experiment` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SampleSize` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `SRMCheck` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `MultipleCorrection` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Variance (`splita.variance`) — M4

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `CUPED` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `CUPAC` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `OutlierHandler` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |

### Sequential (`splita.sequential`) — M5

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `mSPRT` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `GroupSequential` | Y | Y | Y | Y | Y | Y | Y | Y | Y | DONE |
| `EValue` | Y | Y | Y | Y | Y | Y | Y | N | Y | TODO |

### Bandits (`splita.bandits`) — M6

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `ThompsonSampler` | Y | Y | Y | Y | Y | Y | Y | N | Y | TODO |
| `LinTS` | Y | Y | Y | Y | Y | Y | Y | N | Y | TODO |
| `LinUCB` | Y | N | N | N | N | N | N | N | Y | TODO |
| `BayesianStopping` | Y | N | N | N | N | N | N | N | Y | TODO |

### Core — v0.2 additions (M8-M14)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `BayesianExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `QuantileExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `PowerSimulation` | Y | N | N | N | N | N | N | N | Y | TODO |
| `HTEEstimator` | Y | N | N | N | N | N | N | N | Y | TODO |
| `TriggeredExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `InteractionTest` | Y | N | N | N | N | N | N | N | Y | TODO |
| `MultiObjectiveExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `StratifiedExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `CausalForest` | Y | N | N | N | N | N | N | N | Y | TODO |

### Variance — v0.3 additions

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `MultivariateCUPED` | Y | N | N | N | N | N | N | N | Y | TODO |
| `RegressionAdjustment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `AdaptiveWinsorizer` | Y | N | N | N | N | N | N | N | Y | TODO |
| `DoubleML` | Y | N | N | N | N | N | N | N | Y | TODO |

### Sequential — v0.3 additions

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `ConfidenceSequence` | Y | N | N | N | N | N | N | N | Y | TODO |
| `EProcess` | Y | N | N | N | N | N | N | N | Y | TODO |

### Diagnostics (`splita.diagnostics`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `NoveltyCurve` | Y | N | N | N | N | N | N | N | Y | TODO |
| `AATest` | Y | N | N | N | N | N | N | N | Y | TODO |
| `EffectTimeSeries` | Y | N | N | N | N | N | N | N | Y | TODO |
| `MetricSensitivity` | Y | N | N | N | N | N | N | N | Y | TODO |
| `VarianceEstimator` | Y | N | N | N | N | N | N | N | Y | TODO |
| `NonStationaryDetector` | Y | N | N | N | N | N | N | N | Y | TODO |

### Causal (`splita.causal`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `DifferenceInDifferences` | Y | N | N | N | N | N | N | N | Y | TODO |
| `SyntheticControl` | Y | N | N | N | N | N | N | N | Y | TODO |
| `ClusterExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `SwitchbackExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |
| `SurrogateEstimator` | Y | N | N | N | N | N | N | N | Y | TODO |
| `SurrogateIndex` | Y | N | N | N | N | N | N | N | Y | TODO |
| `InterferenceExperiment` | Y | N | N | N | N | N | N | N | Y | TODO |

### Design (`splita.design`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `PairwiseDesign` | Y | N | N | N | N | N | N | N | Y | TODO |

### Governance (`splita.governance`)

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `ExperimentRegistry` | Y | N | N | N | N | N | - | - | Y | TODO |
| `ConflictDetector` | Y | N | N | N | N | N | - | - | Y | TODO |

---

## PLANNED Classes (44) — from PRD v0.4

| Class | Tests | Stats | Maths | TL | QA | Sec | E2E | Audit | Docs | Status |
|-------|-------|-------|-------|-----|-----|-----|-----|-------|------|--------|
| `GuardrailMonitor` | N | N | N | N | N | N | N | N | N | TODO |
| `FlickerDetector` | N | N | N | N | N | N | N | N | N | TODO |
| `TrimmedMeanEstimator` | N | N | N | N | N | N | N | N | N | TODO |
| `OECBuilder` | N | N | N | N | N | N | N | N | N | TODO |
| `PostStratification` | N | N | N | N | N | N | N | N | N | TODO |
| `ClusterBootstrap` | N | N | N | N | N | N | N | N | N | TODO |
| `RandomizationValidator` | N | N | N | N | N | N | N | N | N | TODO |
| `FunnelExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `InterleavingExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `CarryoverDetector` | N | N | N | N | N | N | N | N | N | TODO |
| `RobustMeanEstimator` | N | N | N | N | N | N | N | N | N | TODO |
| `InstrumentalVariables` | N | N | N | N | N | N | N | N | N | TODO |
| `PropensityScoreMatching` | N | N | N | N | N | N | N | N | N | TODO |
| `ResponseAdaptiveRandomization` | N | N | N | N | N | N | N | N | N | TODO |
| `InExperimentVR` | N | N | N | N | N | N | N | N | N | TODO |
| `MetricDecomposition` | N | N | N | N | N | N | N | N | N | TODO |
| `ObjectiveBayesianExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `RiskAwareDecision` | N | N | N | N | N | N | N | N | N | TODO |
| `OptimalProxyMetrics` | N | N | N | N | N | N | N | N | N | TODO |
| `DoublyRobustEstimator` | N | N | N | N | N | N | N | N | N | TODO |
| `SampleSizeReestimation` | N | N | N | N | N | N | N | N | N | TODO |
| `RegressionDiscontinuity` | N | N | N | N | N | N | N | N | N | TODO |
| `NonstationaryAdjustment` | N | N | N | N | N | N | N | N | N | TODO |
| `GeoExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `PredictionPoweredInference` | N | N | N | N | N | N | N | N | N | TODO |
| `EffectTransport` | N | N | N | N | N | N | N | N | N | TODO |
| `TMLE` | N | N | N | N | N | N | N | N | N | TODO |
| `DynamicCausalEffect` | N | N | N | N | N | N | N | N | N | TODO |
| `BayesianExperimentOptimizer` | N | N | N | N | N | N | N | N | N | TODO |
| `AdaptiveEnrichment` | N | N | N | N | N | N | N | N | N | TODO |
| `SurvivalExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `MediationAnalysis` | N | N | N | N | N | N | N | N | N | TODO |
| `PermutationTest` | N | N | N | N | N | N | N | N | N | TODO |
| `MixedEffectsExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `FractionalFactorialDesign` | N | N | N | N | N | N | N | N | N | TODO |
| `ContinuousTreatmentEffect` | N | N | N | N | N | N | N | N | N | TODO |
| `PHackingDetector` | N | N | N | N | N | N | N | N | N | TODO |
| `OfflineEvaluator` | N | N | N | N | N | N | N | N | N | TODO |
| `YEASTSequentialTest` | N | N | N | N | N | N | N | N | N | TODO |
| `MarketplaceExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `BudgetSplitDesign` | N | N | N | N | N | N | N | N | N | TODO |
| `BipartiteExperiment` | N | N | N | N | N | N | N | N | N | TODO |
| `DilutionAnalysis` | N | N | N | N | N | N | N | N | N | TODO |
| `ExperimentationAccelerator` | N | N | N | N | N | N | N | N | N | TODO |

---

## Summary

| Status | Count |
|--------|-------|
| DONE (all 9 gates passed) | 14 |
| TODO — built but not fully reviewed | 27 |
| TODO — not yet built | 44 |
| **Total** | **85** |
