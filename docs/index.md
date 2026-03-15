# splita

**A/B test analysis that is correct by default, informative by design, and composable by construction.**

[![PyPI version](https://img.shields.io/pypi/v/splita)](https://pypi.org/project/splita/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Naareman/splita/ci.yml?label=tests)](https://github.com/Naareman/splita/actions)
[![Coverage](https://img.shields.io/codecov/c/github/Naareman/splita)](https://codecov.io/gh/Naareman/splita)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/splita/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/Naareman/splita/blob/main/LICENSE)

---

## What is splita?

splita is a Python library for A/B test analysis. It provides **88 classes** across 8 modules covering the full experimentation lifecycle: planning, data quality checks, frequentist and Bayesian analysis, sequential testing, variance reduction, causal inference, bandits, and governance.

Everything returns frozen dataclasses with `.to_dict()`. No vendor lock-in, no DataFrames required, no global state.

## Feature highlights

- **Frequentist + Bayesian** -- z-test, t-test, Mann-Whitney, chi-square, bootstrap, Bayesian posterior inference
- **Sequential testing** -- mSPRT, Group Sequential, e-values, Confidence Sequences, YEAST
- **Variance reduction** -- CUPED, CUPAC, Double ML, regression adjustment, 14 methods total
- **Causal inference** -- DiD, Synthetic Control, TMLE, Propensity Matching, 19 classes total
- **Bandits** -- Thompson Sampling, LinUCB, LinTS, offline evaluation
- **Data quality** -- SRM checks, flicker detection, randomization validation
- **Diagnostics** -- novelty curves, carryover detection, p-hacking analysis
- **`explain()` in 4 languages** -- plain-English (and Arabic, etc.) summaries of any result

## Quick start

```python
from splita import Experiment

# Pass control and treatment arrays -- splita auto-detects the metric type
result = Experiment([0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 1, 1]).run()
print(result.significant)    # False
print(result.relative_lift)  # 66.67%
print(result.pvalue)         # 0.1432
```

A more realistic example with continuous data:

```python
from splita import Experiment
import numpy as np

rng = np.random.default_rng(42)
ctrl = rng.normal(25.0, 8.0, size=1000)
trt = rng.normal(26.5, 8.0, size=1000)

result = Experiment(ctrl, trt).run()
print(result.significant)    # True
print(result.relative_lift)  # ~6%
```

Get a plain-English explanation:

```python
from splita import explain
print(explain(result))
```

Generate a full report:

```python
from splita import report
print(report(result))
```

## Dependencies

- **Required:** numpy >= 1.24, scipy >= 1.10
- **Optional:** scikit-learn >= 1.2 (for CUPAC), matplotlib >= 3.5 (for viz), ipywidgets >= 8.0 (for Jupyter widgets)

## Next steps

- [Installation](getting-started/installation.md) -- install splita and optional extras
- [Quick Start tutorial](getting-started/quickstart.md) -- 5-minute walkthrough
- [Core Concepts](getting-started/concepts.md) -- A/B testing fundamentals
- [API Reference](api/index.md) -- all 88 classes at a glance
