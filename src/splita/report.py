"""Self-contained experiment report generation.

Generate HTML or plain-text reports from any combination of splita result
objects.  Reports include summary, data quality (SRM), primary metrics,
and actionable recommendations.

Examples
--------
>>> from splita import Experiment, SRMCheck, report
>>> import numpy as np
>>> ctrl = np.random.binomial(1, 0.10, 5000)
>>> trt  = np.random.binomial(1, 0.12, 5000)
>>> exp_result = Experiment(ctrl, trt).run()
>>> srm_result = SRMCheck([5000, 5000]).run()
>>> html = report(exp_result, srm_result, title="Q1 Conversion Test")
"""

from __future__ import annotations

import html as html_mod
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

from splita.explain import explain

# ─── Result type classification ──────────────────────────────────────

_SRM_TYPES = {"SRMResult"}
_PRIMARY_TYPES = {"ExperimentResult", "BayesianResult"}
_SECONDARY_TYPES = {
    "SampleSizeResult",
    "QuantileResult",
    "StratifiedResult",
    "SurvivalResult",
    "FunnelResult",
    "PermutationResult",
    "HTEResult",
}
_SEQUENTIAL_TYPES = {
    "GSResult",
    "mSPRTResult",
    "CSResult",
    "EValueResult",
    "EProcessResult",
    "YEASTResult",
    "BayesianStoppingResult",
}
_VARIANCE_TYPES = {
    "CorrectionResult",
    "RegressionAdjustmentResult",
    "TrimmedMeanResult",
    "RobustMeanResult",
    "ClusterBootstrapResult",
    "PostStratResult",
    "DoubleMLResult",
    "PPIResult",
    "InExperimentVRResult",
    "NonstationaryAdjResult",
    "VarianceEstimateResult",
}


def _classify(result: Any) -> str:
    """Return the section a result belongs to."""
    name = type(result).__name__
    if name in _SRM_TYPES:
        return "srm"
    if name in _PRIMARY_TYPES:
        return "primary"
    if name in _SEQUENTIAL_TYPES:
        return "sequential"
    if name in _VARIANCE_TYPES:
        return "variance"
    return "secondary"


# ─── Plain-text rendering ────────────────────────────────────────────


def _text_section(heading: str, body: str) -> str:
    """Format a plain-text section."""
    sep = "=" * len(heading)
    return f"{heading}\n{sep}\n{body}\n"


def _result_to_text(result: Any) -> str:
    """Render a single result as plain text."""
    type_name = type(result).__name__
    lines = [f"  [{type_name}]"]

    # Use explain() for supported types
    try:
        explanation = explain(result)
        lines.append(f"  {explanation}")
    except TypeError:  # pragma: no cover
        pass

    # Show key fields
    for f in fields(result):
        val = getattr(result, f.name)
        if isinstance(val, float):
            if abs(val) < 0.0001 and val != 0.0:
                lines.append(f"  {f.name}: {val:.2e}")
            else:
                lines.append(f"  {f.name}: {val:.4f}")
        elif isinstance(val, (list, tuple)) and len(val) > 10:
            lines.append(f"  {f.name}: [{len(val)} items]")
        else:
            lines.append(f"  {f.name}: {val}")

    return "\n".join(lines)


def _render_text(
    results: tuple[Any, ...],
    title: str,
) -> str:
    """Render a full plain-text report."""
    sections: list[str] = []
    sections.append(f"{title}\n{'#' * len(title)}\n")
    sections.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Classify results
    srm = [r for r in results if _classify(r) == "srm"]
    primary = [r for r in results if _classify(r) == "primary"]
    sequential = [r for r in results if _classify(r) == "sequential"]
    variance_results = [r for r in results if _classify(r) == "variance"]
    secondary = [r for r in results if _classify(r) == "secondary"]

    # Summary
    summary_lines = [f"Total results: {len(results)}"]
    if srm:
        all_passed = all(getattr(r, "passed", True) for r in srm)
        summary_lines.append(f"Data quality: {'PASS' if all_passed else 'FAIL'}")
    if primary:
        sig_count = sum(1 for r in primary if getattr(r, "significant", False))
        summary_lines.append(f"Significant results: {sig_count}/{len(primary)}")
    sections.append(_text_section("Summary", "\n".join(summary_lines)))

    # SRM / data quality
    if srm:
        body = "\n\n".join(_result_to_text(r) for r in srm)
        sections.append(_text_section("Data Quality (SRM)", body))

    # Primary
    if primary:
        body = "\n\n".join(_result_to_text(r) for r in primary)
        sections.append(_text_section("Primary Metrics", body))

    # Sequential
    if sequential:
        body = "\n\n".join(_result_to_text(r) for r in sequential)
        sections.append(_text_section("Sequential Tests", body))

    # Variance reduction
    if variance_results:
        body = "\n\n".join(_result_to_text(r) for r in variance_results)
        sections.append(_text_section("Variance Reduction", body))

    # Secondary
    if secondary:
        body = "\n\n".join(_result_to_text(r) for r in secondary)
        sections.append(_text_section("Secondary Metrics", body))

    # Recommendations
    recommendations = _gather_recommendations(results)
    if recommendations:
        body = "\n".join(f"  - {r}" for r in recommendations)
        sections.append(_text_section("Recommendations", body))

    return "\n".join(sections)


# ─── HTML rendering ──────────────────────────────────────────────────

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
       sans-serif; color: #333; max-width: 900px; margin: 0 auto;
       padding: 24px; background: #fafafa; }
h1 { font-size: 24px; margin-bottom: 4px; color: #1a1a2e; }
.subtitle { font-size: 13px; color: #888; margin-bottom: 24px; }
.section { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
           padding: 20px; margin-bottom: 16px; }
.section h2 { font-size: 16px; color: #1a1a2e; margin-bottom: 12px;
              border-bottom: 2px solid #4361ee; padding-bottom: 6px;
              display: inline-block; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 12px; font-weight: 600; margin-left: 8px;
         vertical-align: middle; }
.badge-pass { background: #d4edda; color: #155724; }
.badge-fail { background: #f8d7da; color: #721c24; }
.badge-sig  { background: #cce5ff; color: #004085; }
.badge-ns   { background: #fff3cd; color: #856404; }
table { width: 100%; border-collapse: collapse; font-size: 13px;
        margin-top: 8px; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }
th { font-weight: 600; color: #555; background: #f8f9fa; }
tr:hover { background: #f5f5f5; }
.mono { font-family: 'SF Mono', 'Menlo', monospace; }
.explain { background: #f0f4ff; border-left: 3px solid #4361ee;
           padding: 12px 16px; margin-top: 12px; font-size: 13px;
           line-height: 1.5; border-radius: 0 4px 4px 0; white-space: pre-wrap; }
.rec-list { list-style: none; padding: 0; }
.rec-list li { padding: 8px 12px; margin-bottom: 6px; background: #fff8e1;
               border-left: 3px solid #ffc107; border-radius: 0 4px 4px 0;
               font-size: 13px; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin-top: 12px; }
.stat-card { background: #f8f9fa; border-radius: 6px; padding: 12px 16px;
             text-align: center; }
.stat-card .label { font-size: 11px; color: #888; text-transform: uppercase;
                    letter-spacing: 0.5px; }
.stat-card .value { font-size: 22px; font-weight: 700; color: #1a1a2e;
                    margin-top: 2px; }
"""


def _fmt_val(val: Any) -> str:
    """Format a value for HTML display."""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        if abs(val) < 0.0001 and val != 0.0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    if isinstance(val, (list, tuple)):
        if len(val) > 10:
            return f"[{len(val)} items]"
        return str(val)
    return str(val)


def _result_to_html_table(result: Any) -> str:
    """Render a result as an HTML table."""
    type_name = type(result).__name__
    rows: list[str] = []
    for i, f in enumerate(fields(result)):
        val = getattr(result, f.name)
        bg = ' style="background: #f8f9fa;"' if i % 2 == 0 else ""
        val_str = html_mod.escape(_fmt_val(val))
        rows.append(
            f'<tr{bg}><td>{html_mod.escape(f.name)}</td><td class="mono">{val_str}</td></tr>'
        )
    return (
        f'<h3 style="font-size: 14px; margin-top: 16px; color: #555;">{type_name}</h3>'
        f"<table><thead><tr><th>Field</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _result_explain_html(result: Any) -> str:
    """Get the explain() output wrapped in HTML, or empty string."""
    try:
        text = explain(result)
        return f'<div class="explain">{html_mod.escape(text)}</div>'
    except TypeError:  # pragma: no cover
        return ""


def _render_html(
    results: tuple[Any, ...],
    title: str,
) -> str:
    """Render a full self-contained HTML report."""
    srm = [r for r in results if _classify(r) == "srm"]
    primary = [r for r in results if _classify(r) == "primary"]
    sequential = [r for r in results if _classify(r) == "sequential"]
    variance_results = [r for r in results if _classify(r) == "variance"]
    secondary = [r for r in results if _classify(r) == "secondary"]

    parts: list[str] = []

    # Header
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append(f"<title>{html_mod.escape(title)}</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>{html_mod.escape(title)}</h1>")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f'<p class="subtitle">Generated {ts} by splita</p>')

    # Summary section
    parts.append('<div class="section"><h2>Summary</h2>')
    parts.append('<div class="summary-grid">')
    parts.append(
        f'<div class="stat-card"><div class="label">Results</div>'
        f'<div class="value">{len(results)}</div></div>'
    )
    if srm:
        all_passed = all(getattr(r, "passed", True) for r in srm)
        badge = "badge-pass" if all_passed else "badge-fail"
        label = "PASS" if all_passed else "FAIL"
        parts.append(
            f'<div class="stat-card"><div class="label">Data Quality</div>'
            f'<div class="value"><span class="badge {badge}">{label}</span></div></div>'
        )
    if primary:
        sig_count = sum(1 for r in primary if getattr(r, "significant", False))
        parts.append(
            f'<div class="stat-card"><div class="label">Significant</div>'
            f'<div class="value">{sig_count}/{len(primary)}</div></div>'
        )
    parts.append("</div></div>")

    # SRM section
    if srm:
        parts.append('<div class="section"><h2>Data Quality (SRM)</h2>')
        for r in srm:
            passed = getattr(r, "passed", True)
            badge_cls = "badge-pass" if passed else "badge-fail"
            badge_txt = "PASS" if passed else "FAIL"
            parts.append(f'<span class="badge {badge_cls}">{badge_txt}</span>')
            parts.append(_result_to_html_table(r))
            parts.append(_result_explain_html(r))
        parts.append("</div>")

    # Primary section
    if primary:
        parts.append('<div class="section"><h2>Primary Metrics</h2>')
        for r in primary:
            sig = getattr(r, "significant", None)
            if sig is not None:
                badge_cls = "badge-sig" if sig else "badge-ns"
                badge_txt = "Significant" if sig else "Not significant"
                parts.append(f'<span class="badge {badge_cls}">{badge_txt}</span>')
            parts.append(_result_to_html_table(r))
            parts.append(_result_explain_html(r))
        parts.append("</div>")

    # Sequential section
    if sequential:
        parts.append('<div class="section"><h2>Sequential Tests</h2>')
        for r in sequential:
            parts.append(_result_to_html_table(r))
            parts.append(_result_explain_html(r))
        parts.append("</div>")

    # Variance reduction section
    if variance_results:
        parts.append('<div class="section"><h2>Variance Reduction</h2>')
        for r in variance_results:
            parts.append(_result_to_html_table(r))
            parts.append(_result_explain_html(r))
        parts.append("</div>")

    # Secondary section
    if secondary:
        parts.append('<div class="section"><h2>Secondary Metrics</h2>')
        for r in secondary:
            parts.append(_result_to_html_table(r))
            parts.append(_result_explain_html(r))
        parts.append("</div>")

    # Recommendations
    recommendations = _gather_recommendations(results)
    if recommendations:
        parts.append('<div class="section"><h2>Recommendations</h2>')
        parts.append('<ul class="rec-list">')
        for rec in recommendations:
            parts.append(f"<li>{html_mod.escape(rec)}</li>")
        parts.append("</ul></div>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ─── Recommendations ─────────────────────────────────────────────────


def _gather_recommendations(results: tuple[Any, ...] | list[Any]) -> list[str]:
    """Extract actionable recommendations from results."""
    recs: list[str] = []
    seen: set[str] = set()

    def _add(msg: str) -> None:
        if msg not in seen:
            recs.append(msg)
            seen.add(msg)

    for r in results:
        name = type(r).__name__

        if name == "SRMResult" and not getattr(r, "passed", True):
            _add(
                "Sample Ratio Mismatch detected. Investigate randomization, "
                "bot traffic, and tracking pipeline before trusting results."
            )

        if name == "ExperimentResult":
            if not getattr(r, "significant", False):
                _add(
                    "No significant effect detected. Consider running longer "
                    "or using variance reduction (CUPED/CUPAC)."
                )
            power = getattr(r, "power", 1.0)
            if power < 0.8:
                _add(
                    "Experiment appears underpowered (post-hoc power < 0.80). "
                    "Use SampleSize for prospective power analysis."
                )

        if name == "BayesianResult":
            prob = getattr(r, "prob_b_beats_a", 0.5)
            if 0.05 < prob < 0.95:
                _add(
                    "Bayesian analysis is not yet decisive. "
                    "Consider collecting more data before making a decision."
                )

        if name == "SampleSizeResult":
            days = getattr(r, "days_needed", None)
            if days is not None and days > 30:
                _add(
                    f"Estimated duration is {days} days. Consider increasing "
                    "the MDE or using variance reduction to shorten the test."
                )

    return recs


# ─── Public API ───────────────────────────────────────────────────────


def report(
    *results: Any,
    title: str = "Experiment Report",
    format: str = "html",
) -> str:
    """Generate a self-contained experiment report.

    Takes any number of splita result objects and produces a formatted
    report with sections for data quality, primary metrics, secondary
    metrics, and recommendations.

    Parameters
    ----------
    *results : dataclass
        Any number of splita result objects (e.g. ``ExperimentResult``,
        ``SRMResult``, ``BayesianResult``, ``SampleSizeResult``).
    title : str, default "Experiment Report"
        Report title.
    format : str, default "html"
        Output format: ``"html"`` for self-contained HTML with inline CSS,
        or ``"text"`` for plain text.

    Returns
    -------
    str
        Report as a string (HTML or plain text).

    Raises
    ------
    ValueError
        If no results are provided or format is invalid.
    TypeError
        If any argument is not a dataclass with fields.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.003,
    ...     statistic=2.1, ci_lower=0.007, ci_upper=0.033,
    ...     significant=True, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=5000,
    ...     treatment_n=5000, power=0.82, effect_size=0.15,
    ... )
    >>> html = report(r)
    >>> "Experiment Report" in html
    True
    """
    if not results:
        raise ValueError(
            "`report()` requires at least one result object.\n"
            "  Hint: pass ExperimentResult, BayesianResult, SRMResult, etc."
        )

    fmt = format.lower()
    if fmt not in ("html", "text"):
        raise ValueError(
            f"`format` must be 'html' or 'text', got {format!r}.\n"
            "  Hint: use format='html' for styled output or format='text' "
            "for plain text."
        )

    # Validate that all results are dataclasses
    for i, r in enumerate(results):
        try:
            fields(r)
        except TypeError:
            raise TypeError(
                f"Argument {i} is not a splita result (got {type(r).__name__}).\n"
                "  Hint: pass result objects returned by .run(), not raw data."
            ) from None

    if fmt == "html":
        return _render_html(results, title)
    return _render_text(results, title)
