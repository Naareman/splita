"""LaTeX export for splita result objects.

Generates ``tabular`` and full ``table`` environments suitable for
inclusion in academic papers.

Examples
--------
>>> from splita._types import ExperimentResult
>>> r = ExperimentResult(
...     control_mean=0.10, treatment_mean=0.12,
...     lift=0.02, relative_lift=0.2, pvalue=0.003,
...     statistic=2.97, ci_lower=0.007, ci_upper=0.033,
...     significant=True, alpha=0.05, method="ztest",
...     metric="conversion", control_n=5000,
...     treatment_n=5000, power=0.82, effect_size=0.15,
... )
>>> print(r.to_latex())  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def _fmt_value(val: Any) -> str:
    """Format a value for LaTeX display."""
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


def to_latex_tabular(result: Any) -> str:
    r"""Generate a LaTeX ``tabular`` environment from a splita result.

    Parameters
    ----------
    result : dataclass
        Any splita result object with dataclass fields.

    Returns
    -------
    str
        LaTeX ``tabular`` source code.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.003,
    ...     statistic=2.97, ci_lower=0.007, ci_upper=0.033,
    ...     significant=True, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=5000,
    ...     treatment_n=5000, power=0.82, effect_size=0.15,
    ... )
    >>> tex = to_latex_tabular(r)
    >>> r'\begin{tabular}' in tex
    True
    """
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Field} & \textbf{Value} \\")
    lines.append(r"\midrule")

    for f in fields(result):
        val = getattr(result, f.name)
        name = _escape_latex(f.name)
        formatted = _escape_latex(_fmt_value(val))
        lines.append(f"{name} & {formatted} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def to_latex_table(
    result: Any,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    r"""Generate a full LaTeX ``table`` environment with caption and label.

    Parameters
    ----------
    result : dataclass
        Any splita result object with dataclass fields.
    caption : str or None, default None
        Table caption.  If ``None``, uses the result class name.
    label : str or None, default None
        LaTeX label for cross-referencing (e.g. ``"tab:experiment"``).

    Returns
    -------
    str
        Complete LaTeX ``table`` environment.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.003,
    ...     statistic=2.97, ci_lower=0.007, ci_upper=0.033,
    ...     significant=True, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=5000,
    ...     treatment_n=5000, power=0.82, effect_size=0.15,
    ... )
    >>> tex = to_latex_table(r, caption="A/B Test Results", label="tab:ab")
    >>> r'\caption{A/B Test Results}' in tex
    True
    """
    if caption is None:
        caption = type(result).__name__

    tabular = to_latex_tabular(result)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(tabular)
    lines.append(r"\end{table}")
    return "\n".join(lines)
