# splita — Product Requirements Document

**Version**: 0.1.0
**Date**: 2026-03-14
**Status**: Draft

---

## 1. Vision

splita is a Python library for A/B test analysis that is **correct by default, informative by design, and composable by construction**.

Most A/B testing code in production is either (a) scattered utility functions with no validation, (b) heavyweight platforms that force a specific workflow, or (c) academic implementations that are correct but hostile to use. splita occupies the gap: a library that a data scientist can `pip install` and trust to give correct results with informative feedback at every step.

**One-liner**: scipy.stats for A/B testing — correct, composable, zero-opinion on your data stack.

---

## 2. Target Users

| Persona | Needs | How splita helps |
|---------|-------|-----------------|
| **Data Scientist at a startup** | Run A/B tests without a platform | `Experiment(ctrl, trt).run()` — one line, correct result |
| **Experimentation Platform engineer** | Reliable statistical primitives to build on | Composable functions with dataclass outputs, no side effects |
| **Growth/Product Analyst** | Understand if a test is trustworthy | Informative error messages, SRM checks, power analysis |
| **ML Engineer** | Bandit algorithms for real-time optimization | Thompson Sampling, LinTS with streaming API |

---

## 3. Design Principles

### 3.1 Python Structure, R Soul

The package follows Python packaging conventions (pyproject.toml, src layout, pytest, type hints) but adopts the **user experience philosophy of R's tidyverse**:

1. **Discoverable naming**: Related functions share a common prefix. `SampleSize.for_proportion()`, `SampleSize.for_mean()`, `SampleSize.for_ratio()` — type `SampleSize.for_` and autocomplete shows you everything.

2. **Informative errors** (3-part structure):
   ```
   ValueError: `alpha` must be in (0, 1), got 1.5.
     Detail: alpha=1.5 means a 150% false positive rate, which is not meaningful.
     Hint: typical values are 0.05, 0.01, or 0.10.
   ```
   - **Problem**: what went wrong (use "must" or "can't")
   - **Detail**: the actual bad value and why it's wrong
   - **Hint** (optional): suggested fix, only when confident

3. **String enums over booleans**: `alternative="two-sided"` not `two_sided=True`. Self-documenting, extensible, no double negatives.

4. **Type stability**: Output type is predictable from input types alone, never from input values.

5. **No magical defaults**: `None` default + explicit handling in body, never hidden behavior changes.

6. **Consistency is the highest virtue**: Same patterns everywhere — naming, arg order, return types, error format. Learn one function, predict all others.

### 3.2 Technical Principles (from API spec)

7. **Type annotations** on every parameter and return value (Python 3.10+ `X | Y` syntax).
8. **NumPy docstrings** (numpydoc format) on all public APIs.
9. **Frozen dataclasses** for all results — attribute access, IDE completion, `.to_dict()` for serialization.
10. **Fail loudly on bad inputs** — never silently coerce or drop data without a warning.
11. **`random_state`** follows NumPy Generator protocol (`int | np.random.Generator | None`).
12. **`array_like` inputs, numpy internals** — accept list, tuple, ndarray, pd.Series; return ndarray.
13. **Keyword-only** config arguments (after positional data arguments).
14. **Sensible defaults** matching industry standard practice (alpha=0.05, power=0.80, two-sided).
15. **sklearn API** for stateful transformers: `fit()` / `transform()` / `fit_transform()`.
16. **Streaming API** for online algorithms: `update()` / `recommend()` / `result()`.
17. **No global state**, no module-level side effects, all randomness local via `random_state`.
18. **Pandas interop**: accept Series, return numpy. Never return DataFrame from statistical functions.

---

## 4. Architecture

### 4.1 Module Structure

```
splita/
├── core/
│   ├── experiment.py      # Experiment class
│   ├── sample_size.py     # SampleSize class
│   ├── srm.py             # SRMCheck class
│   └── correction.py      # MultipleCorrection class
├── variance/
│   ├── cuped.py           # CUPED class
│   ├── cupac.py           # CUPAC class
│   └── outliers.py        # OutlierHandler class
├── sequential/
│   ├── msprt.py           # mSPRT class
│   ├── group_sequential.py # GroupSequential class
│   └── evalue.py          # EValue (TODO)
├── bandits/
│   ├── thompson.py        # ThompsonSampler class
│   ├── lints.py           # LinTS class
│   ├── linucb.py          # LinUCB (TODO)
│   └── bayesian_stopping.py # BayesianStopping (TODO)
├── _types.py              # All result dataclasses
├── _validation.py         # Shared validation helpers (error message formatting)
└── _utils.py              # Array conversion, RNG handling
```

### 4.2 Dependency Policy

| Dependency | Version | Reason |
|-----------|---------|--------|
| numpy | >=1.24 | Core array operations |
| scipy | >=1.10 | Statistical distributions, special functions |

**Optional** (not required for core):
| Dependency | When needed |
|-----------|-------------|
| pandas | Only for Series detection in input validation |
| scikit-learn | Only for CUPAC (user provides estimator) |

No other dependencies. Ever.

### 4.3 Result Dataclasses

Every public function returns a frozen dataclass with:
- All computed fields as typed attributes
- `.to_dict() -> dict[str, Any]` for serialization
- `__repr__` that prints a human-readable summary (not just field dump)

```python
@dataclass(frozen=True)
class ExperimentResult:
    control_mean: float
    treatment_mean: float
    lift: float
    relative_lift: float
    pvalue: float
    statistic: float
    ci_lower: float
    ci_upper: float
    significant: bool
    alpha: float
    method: str
    control_n: int
    treatment_n: int
    power: float       # post-hoc
    effect_size: float  # Cohen's d or h

    def to_dict(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...  # pretty summary
```

---

## 5. API Specification

> Full parameter contracts, types, defaults, and examples are in `splita_api_spec.docx`. This section summarizes the public API surface.

### 5.1 `splita.core`

| Class | Key Methods | Purpose |
|-------|------------|---------|
| `Experiment` | `__init__(control, treatment, ...)`, `.run()` | Run a statistical test, get ExperimentResult |
| `SampleSize` | `.for_proportion()`, `.for_mean()`, `.for_ratio()`, `.mde_for_proportion()`, `.duration()` | Power analysis and experiment planning |
| `SRMCheck` | `__init__(observed, ...)`, `.run()` | Detect sample ratio mismatch |
| `MultipleCorrection` | `__init__(pvalues, ...)`, `.run()` | BH / Bonferroni / Holm correction |

### 5.2 `splita.variance`

| Class | Key Methods | Purpose |
|-------|------------|---------|
| `CUPED` | `.fit_transform(ctrl, trt, ctrl_pre, trt_pre)` | Variance reduction via pre-experiment covariates |
| `CUPAC` | `.fit_transform(ctrl, trt, X_ctrl, X_trt)` | ML-based variance reduction |
| `OutlierHandler` | `.fit_transform(ctrl, trt)` | Outlier capping (winsorize/trim/IQR/clustering) |

### 5.3 `splita.sequential`

| Class | Key Methods | Purpose |
|-------|------------|---------|
| `mSPRT` | `.update(ctrl_obs, trt_obs)`, `.result()` | Always-valid p-values, continuous monitoring |
| `GroupSequential` | `.boundary()`, `.test(statistics, info_fractions)` | Planned interim analyses with alpha-spending |
| `EValue` | TODO | — |

### 5.4 `splita.bandits`

| Class | Key Methods | Purpose |
|-------|------------|---------|
| `ThompsonSampler` | `.update(arm, reward)`, `.recommend()`, `.result()` | Multi-armed bandit |
| `LinTS` | `.update(arm, ctx, reward)`, `.recommend(ctx)` | Contextual bandit (linear) |
| `LinUCB` | TODO | — |
| `BayesianStopping` | TODO | — |

---

## 6. Milestones

### Milestone 1: Project Scaffold + Core Types
**Deliverables:**
- Project structure (pyproject.toml, src layout, CI config)
- `_types.py` — all result dataclasses
- `_validation.py` — error formatting helpers (3-part structure)
- `_utils.py` — array conversion, RNG handling
- CLAUDE.md with project conventions
- Tests for all utilities and dataclasses

**Review gate**: Tech Lead, QA/QC

### Milestone 2: `splita.core` — Experiment + SampleSize
**Deliverables:**
- `Experiment` class with all 6 test methods (ztest, ttest, mannwhitney, chisquare, delta, bootstrap)
- `SampleSize` class with all 5 methods
- Full test suite with hypothetical A/B scenarios
- Docstrings with examples

**Review gate**: Statistics Expert, Tech Lead, QA/QC

### Milestone 3: `splita.core` — SRMCheck + MultipleCorrection
**Deliverables:**
- `SRMCheck` class
- `MultipleCorrection` class (BH, Bonferroni, Holm, BY)
- Tests including edge cases (very small samples, all-significant, none-significant)

**Review gate**: Statistics Expert, Tech Lead, QA/QC

### Milestone 4: `splita.variance` — CUPED, CUPAC, OutlierHandler
**Deliverables:**
- All 3 variance reduction classes
- Integration tests: OutlierHandler → CUPED → Experiment pipeline
- Variance reduction benchmarks (verify 20-65% reduction on synthetic data)

**Review gate**: Statistics Expert, Tech Lead, QA/QC

### Milestone 5: `splita.sequential` — mSPRT + GroupSequential
**Deliverables:**
- `mSPRT` with streaming API
- `GroupSequential` with OBF/Pocock/Kim-DeMets spending functions
- `EValue` stub (TODO)
- Simulation tests: verify Type I error control under peeking

**Review gate**: Statistics Expert, Tech Lead, QA/QC

### Milestone 6: `splita.bandits` — ThompsonSampler + LinTS
**Deliverables:**
- `ThompsonSampler` (Bernoulli, Gaussian, Poisson)
- `LinTS` contextual bandit
- `LinUCB` and `BayesianStopping` stubs (TODO)
- Regret benchmarks on synthetic problems

**Review gate**: Statistics Expert, Tech Lead, QA/QC

### Milestone 7: Integration, Documentation, Polish
**Deliverables:**
- End-to-end scenario tests (full experiment lifecycle)
- User guide / vignettes (problem-oriented, not function-oriented)
- Performance benchmarks
- README with quickstart
- PyPI-ready packaging

**Review gate**: All reviewers (Statistics, Tech Lead, QA/QC)

---

## 7. Testing Strategy

### 7.1 Unit Tests
- Every public method has at least one test
- Validation: every `ValueError` / `TypeError` path is tested
- Edge cases: empty arrays, single observation, all-identical values, NaN handling

### 7.2 Statistical Correctness Tests
- Compare results against scipy.stats, statsmodels, and known analytical solutions
- Property-based tests: e.g., increasing sample size should decrease required MDE
- Calibration tests: run 10,000 simulations, verify Type I error rate is within [0.04, 0.06] for alpha=0.05

### 7.3 Scenario Tests (Hypothetical A/B Tests)
Real-world-inspired scenarios that exercise the full pipeline:

| Scenario | Modules Used |
|----------|-------------|
| E-commerce conversion test (10% baseline, detect +1pp) | SampleSize → Experiment → MultipleCorrection |
| Revenue per user with outliers | OutlierHandler → CUPED → Experiment |
| SRM detection on biased randomizer | SRMCheck |
| Weekly peeking on 4-week experiment | GroupSequential |
| Always-on monitoring dashboard | mSPRT (streaming updates) |
| Email subject line optimization | ThompsonSampler |
| Personalized pricing | LinTS |

### 7.4 Performance Tests
- `Experiment.run()` on 1M observations: target < 100ms
- `mSPRT.update()` per batch: target < 1ms
- `ThompsonSampler.recommend()`: target < 0.1ms

---

## 8. Performance Considerations

- Vectorized NumPy operations everywhere — no Python loops over data
- Pre-allocate arrays for bootstrap resampling
- Use scipy.stats compiled distributions (not hand-rolled PDFs)
- Lazy imports: `sklearn` only imported when CUPAC is instantiated
- Consider numba JIT for bootstrap inner loop if benchmarks warrant it

---

## 9. Out of Scope (v0.1.0)

- Web UI or dashboard
- Database integration
- Automatic experiment tracking
- Bayesian A/B testing (full posterior inference) — future version
- Multi-armed bandit with non-stationary rewards
- Network effects / interference detection
- EValue, LinUCB, BayesianStopping (deferred to v0.2.0)

---

## 10. Design Decisions (Resolved)

> Challenges raised against the original API spec, resolved 2026-03-14:

1. **`Experiment.method="auto"` — keep simple, warn on skew**: Auto follows the spec's deterministic rule (binary→ztest, continuous→ttest, ratio→delta). No auto-switching based on data shape. Emit `RuntimeWarning` when skewness is high, suggesting `method="mannwhitney"` or `method="bootstrap"`.

2. **`CUPED.covariate` — drop `"pre_metric"`**: Only `"auto"` and `"custom"` are valid values. `"pre_metric"` is removed to avoid redundant naming. Document that `"auto"` uses pre-experiment metric values.

3. **`SampleSize.duration()` — both standalone and chained**: Keep the standalone `SampleSize.duration(n_required, daily_users, ...)` classmethod. Also add `.duration(daily_users, ...)` to `SampleSizeResult` for chaining: `SampleSize.for_proportion(...).duration(daily_users=500)`.

4. **Mann-Whitney CI — Hodges-Lehmann by default**: Use the Hodges-Lehmann estimator with Moses CI (deterministic, no randomness) as default. Allow `ci_method="bootstrap"` override for users who prefer it. This avoids requiring `n_bootstrap`/`random_state` for a non-parametric test.

5. **`OutlierHandler.method="clustering"` — deferred to v0.2.0**: Ship v0.1.0 with `"winsorize"`, `"trim"`, `"iqr"` only. `"clustering"` raises a clear error with hint. Added to TODO alongside EValue, LinUCB, BayesianStopping.

6. **`mSPRT.tau` auto-tuning — auto-tune with transparency**: Auto-tuning from first batch is the default. Emit info message showing computed tau value. Store as `tau_` fitted attribute for inspection/reuse. Document that explicit tau is recommended for production reproducibility.
