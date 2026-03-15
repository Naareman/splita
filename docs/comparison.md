# Comparison

How does splita compare to other experimentation tools?

## Feature comparison

| | **splita** | **GrowthBook** | **Eppo** | **statsmodels** |
|---|---|---|---|---|
| **Type** | Python library | SaaS + SDK | SaaS + SDK | Python library |
| **Pricing** | Free (MIT) | Free tier / paid | Paid | Free (BSD) |
| **Self-hosted** | Yes (it's a library) | Yes (complex) | No | Yes (it's a library) |
| **Data stays local** | Always | Self-hosted option | No | Always |
| **Vendor lock-in** | None | Moderate | High | None |

## Statistical capabilities

| | **splita** | **GrowthBook** | **Eppo** | **statsmodels** |
|---|---|---|---|---|
| **Frequentist A/B** | z-test, t-test, Mann-Whitney, chi-square, delta, bootstrap | z-test, t-test | z-test, t-test | z-test, t-test |
| **Bayesian A/B** | Built-in | Built-in | Built-in | No |
| **Sequential testing** | mSPRT, YEAST, e-values, Group Sequential, Confidence Sequences | Sequential | Sequential | No |
| **Variance reduction** | CUPED, CUPAC, Double ML, 14 methods total | CUPED | CUPED | Manual |
| **Causal inference** | DiD, Synthetic Control, TMLE, PSM, 19 classes | No | No | DiD (basic) |
| **Bandits** | Thompson, LinUCB, LinTS, offline eval | Multi-armed bandits | Bandits | No |
| **Multiple testing** | BH, Bonferroni, Holm, BY | Limited | Limited | BH, Bonferroni |
| **Power analysis** | Built-in + Monte Carlo | Limited | No | Built-in |
| **Heterogeneous effects** | HTE, CausalForest, InteractionTest | No | No | No |
| **Diagnostics** | 10 classes (SRM, novelty, flicker, etc.) | SRM | SRM | No |

## Developer experience

| | **splita** | **GrowthBook** | **Eppo** | **statsmodels** |
|---|---|---|---|---|
| **Lines for a z-test** | 3 | ~20 (SDK + config) | ~20 (SDK + config) | 8 |
| **Dependencies** | numpy + scipy only | Node.js / Docker | SaaS | numpy + scipy + pandas |
| **Result format** | Frozen dataclasses, `.to_dict()` | JSON API | JSON API | Mixed (arrays, objects) |
| **Jupyter-native** | Yes (HTML repr, widgets) | No | No | Partial |
| **`explain()` in 4 languages** | Yes | No | No | No |
| **LaTeX export** | Yes | No | No | Yes |
| **REST API** | `serve()` one-liner | Built-in | Built-in | No |
| **Plugin system** | `register_method()` | Feature flags | Feature flags | No |

## When to use what

### Use splita when

- You want a **Python-native** experimentation toolkit
- You need **correct statistical defaults** without configuration
- You need **causal inference**, **sequential testing**, or **bandits** beyond basic t-tests
- Your data must stay on **your infrastructure** (no vendor lock-in)
- You want to **compose** analysis steps (outlier handling + CUPED + experiment)

### Use GrowthBook or Eppo when

- You need a **full-stack feature flagging + experimentation platform** with a web UI
- You need **team collaboration** features (dashboards, approvals, audit logs)
- You are okay with SaaS or self-hosted infrastructure overhead
- Your team includes non-technical stakeholders who need a GUI

### Use statsmodels when

- You need **general-purpose statistics** (regression, time series, GLMs) beyond A/B testing
- You are already using pandas DataFrames throughout your pipeline
- You want access to classical econometric models

## Code comparison

### Z-test for proportions

=== "splita"

    ```python
    from splita import Experiment
    result = Experiment(ctrl, trt).run()
    print(result.pvalue)
    ```

=== "statsmodels"

    ```python
    from statsmodels.stats.proportion import proportions_ztest
    import numpy as np
    count = np.array([ctrl.sum(), trt.sum()])
    nobs = np.array([len(ctrl), len(trt)])
    stat, pval = proportions_ztest(count, nobs)
    print(pval)
    ```

=== "GrowthBook"

    ```python
    # Requires SDK setup, feature flag config, and data warehouse integration
    # ~20 lines of configuration before you can run a test
    ```

### CUPED variance reduction

=== "splita"

    ```python
    from splita.variance import CUPED
    cuped = CUPED()
    ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
    ```

=== "statsmodels"

    ```python
    # Manual implementation required:
    # 1. Compute covariance between pre and post
    # 2. Compute theta = cov(Y, X) / var(X)
    # 3. Adjust: Y_adj = Y - theta * (X - mean(X))
    # ~15 lines of numpy/scipy code
    ```

=== "GrowthBook / Eppo"

    ```
    CUPED is a platform-level feature, not user-accessible code.
    Configuration via web UI.
    ```
