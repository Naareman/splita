"""Visual power analysis report generation.

Provides :func:`power_report` which generates formatted tables showing
power at different sample sizes, MDE at different sample sizes, and
duration estimates at different traffic levels.

Examples
--------
>>> from splita import power_report
>>> txt = power_report(0.10, format="text")
>>> "Power Analysis" in txt
True
"""

from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm


def _z_alpha(alpha: float, alternative: str = "two-sided") -> float:
    """Critical z-value for the given alpha and sidedness."""
    if alternative == "two-sided":
        return float(norm.ppf(1.0 - alpha / 2.0))
    return float(norm.ppf(1.0 - alpha))


def _power_for_n(
    baseline: float,
    mde: float,
    n: int,
    alpha: float,
    metric: str,
) -> float:
    """Compute power for a given sample size per variant."""
    za = _z_alpha(alpha)
    if metric == "conversion":
        p1, p2 = baseline, baseline + mde
        se0 = math.sqrt(2.0 * ((p1 + p2) / 2.0) * (1.0 - (p1 + p2) / 2.0) / n)
        se1 = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n)
    else:
        # Continuous: assume baseline is the mean and mde is absolute,
        # with std dev approximated as baseline (or 1 if baseline is 0).
        sd = baseline if baseline > 0 else 1.0
        se0 = sd * math.sqrt(2.0 / n)
        se1 = se0
    if se1 == 0:
        return 1.0
    zb = (abs(mde) - za * se0) / se1
    return float(norm.cdf(zb))


def _n_for_power(
    baseline: float,
    mde: float,
    alpha: float,
    target_power: float,
    metric: str,
) -> int:
    """Compute required n per variant for a target power level."""
    za = _z_alpha(alpha)
    zb = float(norm.ppf(target_power))
    if metric == "conversion":
        p1, p2 = baseline, baseline + mde
        p_bar = (p1 + p2) / 2.0
        se0_unit = math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        se1_unit = math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    else:
        sd = baseline if baseline > 0 else 1.0
        se0_unit = sd * math.sqrt(2.0)
        se1_unit = se0_unit
    if mde == 0:
        return 0
    numerator = (za * se0_unit + zb * se1_unit) ** 2
    return math.ceil(numerator / (mde**2))


def _mde_for_n(
    baseline: float,
    n: int,
    alpha: float,
    power: float,
    metric: str,
) -> float:
    """Compute the MDE for a given n per variant via binary search."""
    # Binary search for MDE in a reasonable range
    lo, hi = 1e-8, baseline * 2.0 if baseline > 0 else 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        p = _power_for_n(baseline, mid, n, alpha, metric)
        if p < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def power_report(
    baseline: float,
    *,
    mde_range: list[float] | None = None,
    n_range: list[int] | None = None,
    metric: str = "conversion",
    alpha: float = 0.05,
    format: Literal["text", "html"] = "text",
) -> str:
    """Generate a power analysis report with tables and recommendations.

    Parameters
    ----------
    baseline : float
        Baseline metric value (e.g., 0.10 for 10% conversion rate).
    mde_range : list of float or None, default None
        List of MDE values to analyse. If ``None``, auto-generated as
        1%, 2%, 5%, 10%, 20% relative to baseline.
    n_range : list of int or None, default None
        List of per-variant sample sizes to analyse. If ``None``,
        auto-generated from 500 to 50000.
    metric : str, default 'conversion'
        Metric type: ``'conversion'`` or ``'continuous'``.
    alpha : float, default 0.05
        Significance level.
    format : {'text', 'html'}, default 'text'
        Output format.

    Returns
    -------
    str
        Formatted report (plain text or HTML).

    Examples
    --------
    >>> txt = power_report(0.10, format="text")
    >>> "Power Analysis" in txt
    True
    """
    if mde_range is None:
        if metric == "conversion":
            mde_range = [baseline * r for r in [0.01, 0.02, 0.05, 0.10, 0.20]]
        else:
            mde_range = [baseline * r for r in [0.01, 0.02, 0.05, 0.10, 0.20]]
        # Filter out zero or negative MDEs
        mde_range = [m for m in mde_range if m > 0]

    if n_range is None:
        n_range = [500, 1000, 2000, 5000, 10000, 20000, 50000]

    if format == "html":
        return _html_report(baseline, mde_range, n_range, metric, alpha)
    return _text_report(baseline, mde_range, n_range, metric, alpha)


def _text_report(
    baseline: float,
    mde_range: list[float],
    n_range: list[int],
    metric: str,
    alpha: float,
) -> str:
    """Generate a plain-text power analysis report."""
    lines: list[str] = []
    sep = "-" * 72

    lines.append("=" * 72)
    lines.append("Power Analysis Report")
    lines.append("=" * 72)
    lines.append(f"  Baseline:  {baseline:.4f}")
    lines.append(f"  Metric:    {metric}")
    lines.append(f"  Alpha:     {alpha}")
    lines.append("")

    # Table 1: Power at different N x MDE combinations
    lines.append(sep)
    lines.append("Table 1: Statistical Power (N per variant x MDE)")
    lines.append(sep)

    header = f"{'N':>10}"
    for mde in mde_range:
        header += f"  MDE={mde:.4f}"
    lines.append(header)
    lines.append("-" * len(header))

    for n in n_range:
        row = f"{n:>10}"
        for mde in mde_range:
            pwr = _power_for_n(baseline, mde, n, alpha, metric)
            row += f"  {pwr:>11.1%}"
        lines.append(row)

    lines.append("")

    # Table 2: Required sample size for 80% power at each MDE
    lines.append(sep)
    lines.append("Table 2: Required N per Variant (80% power)")
    lines.append(sep)
    lines.append(f"{'MDE':>12}  {'N per variant':>15}  {'N total':>12}")
    lines.append("-" * 45)

    for mde in mde_range:
        n_req = _n_for_power(baseline, mde, alpha, 0.80, metric)
        lines.append(f"{mde:>12.4f}  {n_req:>15,}  {2 * n_req:>12,}")

    lines.append("")

    # Table 3: Duration estimates at different traffic levels
    traffic_levels = [100, 500, 1000, 5000, 10000]
    lines.append(sep)
    lines.append("Table 3: Experiment Duration (days, 80% power)")
    lines.append(sep)

    header2 = f"{'MDE':>12}"
    for daily in traffic_levels:
        header2 += f"  {daily:>8}/day"
    lines.append(header2)
    lines.append("-" * len(header2))

    for mde in mde_range:
        n_req = _n_for_power(baseline, mde, alpha, 0.80, metric)
        total = 2 * n_req
        row2 = f"{mde:>12.4f}"
        for daily in traffic_levels:
            days = math.ceil(total / daily) if daily > 0 else 0
            row2 += f"  {days:>11}"
        lines.append(row2)

    lines.append("")
    lines.append("=" * 72)

    # Recommendation
    default_mde = mde_range[len(mde_range) // 2] if mde_range else 0
    if default_mde > 0:
        n80 = _n_for_power(baseline, default_mde, alpha, 0.80, metric)
        lines.append(
            f"Recommendation: For a {default_mde:.4f} absolute MDE, "
            f"you need {n80:,} users per variant ({2 * n80:,} total)."
        )

    return "\n".join(lines)


def _html_report(
    baseline: float,
    mde_range: list[float],
    n_range: list[int],
    metric: str,
    alpha: float,
) -> str:
    """Generate an HTML power analysis report."""
    parts: list[str] = []
    parts.append(
        '<div style="font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto;">'
    )
    parts.append("<h2>Power Analysis Report</h2>")
    parts.append(
        f"<p><b>Baseline:</b> {baseline:.4f} | <b>Metric:</b> {metric} | <b>Alpha:</b> {alpha}</p>"
    )

    # Table 1
    parts.append("<h3>Statistical Power (N per variant x MDE)</h3>")
    parts.append('<table style="border-collapse: collapse; width: 100%; font-size: 13px;">')
    parts.append("<tr><th>N</th>")
    for mde in mde_range:
        parts.append(f"<th>MDE={mde:.4f}</th>")
    parts.append("</tr>")

    for n in n_range:
        parts.append(f"<tr><td>{n:,}</td>")
        for mde in mde_range:
            pwr = _power_for_n(baseline, mde, n, alpha, metric)
            color = "#28a745" if pwr >= 0.8 else "#dc3545" if pwr < 0.5 else "#ffc107"
            parts.append(f'<td style="color: {color}; font-weight: 600;">{pwr:.1%}</td>')
        parts.append("</tr>")
    parts.append("</table>")

    # Table 2
    parts.append("<h3>Required N per Variant (80% power)</h3>")
    parts.append('<table style="border-collapse: collapse; width: 100%; font-size: 13px;">')
    parts.append("<tr><th>MDE</th><th>N per variant</th><th>N total</th></tr>")
    for mde in mde_range:
        n_req = _n_for_power(baseline, mde, alpha, 0.80, metric)
        parts.append(f"<tr><td>{mde:.4f}</td><td>{n_req:,}</td><td>{2 * n_req:,}</td></tr>")
    parts.append("</table>")

    parts.append("</div>")
    return "\n".join(parts)
