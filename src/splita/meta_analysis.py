"""Combine results from multiple experiments via meta-analysis.

Provides :func:`meta_analysis` which pools effect estimates using either
fixed-effects (inverse-variance weighting) or random-effects
(DerSimonian-Laird) methodology.

Examples
--------
>>> from splita import meta_analysis
>>> r = meta_analysis([0.05, 0.03, 0.07], [0.02, 0.01, 0.03])
>>> r.method
'random'
"""

from __future__ import annotations

from collections.abc import Sequence

from scipy.stats import chi2, norm

from splita._types import MetaAnalysisResult


def meta_analysis(
    effects: Sequence[float],
    standard_errors: Sequence[float],
    *,
    method: str = "random",
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> MetaAnalysisResult:
    """Combine results from multiple experiments.

    Parameters
    ----------
    effects : sequence of float
        Point estimates from each study.
    standard_errors : sequence of float
        Standard errors of each point estimate.
    method : {'fixed', 'random'}, default 'random'
        ``'fixed'`` uses inverse-variance weighting.
        ``'random'`` uses the DerSimonian-Laird estimator with
        between-study heterogeneity.
    labels : list of str or None, default None
        Study labels for display.
    alpha : float, default 0.05
        Significance level for the confidence interval.

    Returns
    -------
    MetaAnalysisResult
        Pooled effect, standard error, p-value, confidence interval,
        heterogeneity statistics, and per-study weights.

    Raises
    ------
    ValueError
        If inputs are invalid (mismatched lengths, fewer than 2 studies,
        non-positive standard errors, or invalid method).

    Examples
    --------
    >>> r = meta_analysis([0.05, 0.03, 0.07], [0.02, 0.01, 0.03])
    >>> r.method
    'random'
    >>> r.combined_effect > 0
    True
    """
    effects_list = list(effects)
    se_list = list(standard_errors)

    # Validation
    if len(effects_list) != len(se_list):
        raise ValueError(
            "`effects` and `standard_errors` must have the same length, "
            f"got {len(effects_list)} and {len(se_list)}.\n"
            "  Detail: each study needs both an effect estimate and a "
            "standard error.\n"
            "  Hint: check that you passed the correct arrays."
        )
    if len(effects_list) < 2:
        raise ValueError(
            "Meta-analysis requires at least 2 studies, "
            f"got {len(effects_list)}.\n"
            "  Detail: pooling a single study is not meaningful.\n"
            "  Hint: use the raw result directly for a single experiment."
        )
    for i, se in enumerate(se_list):
        if se <= 0:
            raise ValueError(
                f"`standard_errors[{i}]` must be positive, got {se}.\n"
                "  Detail: a non-positive standard error is not valid.\n"
                "  Hint: ensure all standard errors are > 0."
            )
    if method not in ("fixed", "random"):
        raise ValueError(
            f"`method` must be 'fixed' or 'random', got {method!r}.\n"
            "  Detail: only inverse-variance (fixed) and DerSimonian-Laird "
            "(random) are supported.\n"
            "  Hint: use method='fixed' or method='random'."
        )
    if labels is not None and len(labels) != len(effects_list):
        raise ValueError(
            f"`labels` length ({len(labels)}) must match number of "
            f"studies ({len(effects_list)}).\n"
            "  Detail: each study needs a label.\n"
            "  Hint: pass one label per study or leave labels=None."
        )

    k = len(effects_list)

    # Inverse-variance weights for fixed effects
    inv_var = [1.0 / (se**2) for se in se_list]
    w_total = sum(inv_var)
    fixed_weights = [w / w_total for w in inv_var]
    fixed_effect = sum(e * w for e, w in zip(effects_list, fixed_weights, strict=False))

    # Cochran's Q
    q_stat = sum(w * (e - fixed_effect) ** 2 for e, w in zip(effects_list, inv_var, strict=False))
    q_df = k - 1
    q_pvalue = float(1.0 - chi2.cdf(q_stat, df=q_df)) if q_df > 0 else 1.0

    # I-squared
    i_squared = max(0.0, (q_stat - q_df) / q_stat) if q_stat > 0 else 0.0

    if method == "fixed":
        combined_effect = fixed_effect
        combined_se = 1.0 / (w_total**0.5)
        study_weights = fixed_weights
    else:
        # DerSimonian-Laird tau-squared
        c = w_total - sum(w**2 for w in inv_var) / w_total
        tau_sq = max(0.0, (q_stat - q_df) / c) if c > 0 else 0.0

        # Random-effects weights
        re_weights_raw = [1.0 / (se**2 + tau_sq) for se in se_list]
        re_total = sum(re_weights_raw)
        study_weights = [w / re_total for w in re_weights_raw]
        combined_effect = sum(e * w for e, w in zip(effects_list, study_weights, strict=False))
        combined_se = 1.0 / (re_total**0.5)

    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    ci_lower = combined_effect - z_crit * combined_se
    ci_upper = combined_effect + z_crit * combined_se

    z_stat = combined_effect / combined_se if combined_se > 0 else 0.0
    pvalue = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

    return MetaAnalysisResult(
        combined_effect=float(combined_effect),
        combined_se=float(combined_se),
        pvalue=float(pvalue),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        heterogeneity_pvalue=float(q_pvalue),
        i_squared=float(i_squared),
        method=method,
        study_weights=[float(w) for w in study_weights],
        labels=labels,
    )
