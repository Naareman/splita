"""Plotting functions for A/B test results.

All functions accept an optional ``ax`` parameter for composability
and return the parent :class:`matplotlib.figure.Figure`.

matplotlib is an optional dependency.  If not installed, a clear
``ImportError`` is raised on first use.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_matplotlib():
    """Lazy-import matplotlib and return (plt, Figure)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError(
            "splita.viz requires matplotlib. Install with: pip install matplotlib"
        ) from None
    return plt, Figure


def _resolve_ax(ax: Any | None):
    """Return (fig, ax), creating a new figure if *ax* is None."""
    plt, _ = _get_matplotlib()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Forest plot
# ---------------------------------------------------------------------------


def forest_plot(
    results: list,
    labels: list[str] | None = None,
    ax: Any | None = None,
) -> Any:
    """Horizontal CI bars with point estimates for multiple metrics.

    Parameters
    ----------
    results : list[ExperimentResult]
        One result per metric.
    labels : list[str] | None
        Y-axis labels.  Defaults to ``"Metric 0"``, ``"Metric 1"``, etc.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _resolve_ax(ax)

    if labels is None:
        labels = [f"Metric {i}" for i in range(len(results))]

    y_positions = np.arange(len(results))

    for i, (result, _label) in enumerate(zip(results, labels, strict=False)):
        color = "green" if result.significant else "gray"
        ci_lower = result.ci_lower
        ci_upper = result.ci_upper
        lift = result.lift

        ax.errorbar(
            lift,
            i,
            xerr=[[lift - ci_lower], [ci_upper - lift]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=6,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Lift")
    ax.set_title("Forest Plot")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Effect over time
# ---------------------------------------------------------------------------


def effect_over_time(
    time_series_result: Any,
    ax: Any | None = None,
) -> Any:
    """Cumulative lift with CI band over time.

    Parameters
    ----------
    time_series_result : EffectTimeSeriesResult
        Result containing ``time_points`` with per-timestamp data.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _resolve_ax(ax)

    points = time_series_result.time_points
    timestamps = [p["timestamp"] for p in points]
    lifts = [p["cumulative_lift"] for p in points]
    ci_lower = [p["ci_lower"] for p in points]
    ci_upper = [p["ci_upper"] for p in points]

    ax.plot(timestamps, lifts, color="steelblue", marker="o", markersize=4, linewidth=1.5)
    ax.fill_between(timestamps, ci_lower, ci_upper, alpha=0.2, color="steelblue")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Lift")
    ax.set_title("Effect Over Time")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Power curve
# ---------------------------------------------------------------------------


def power_curve(
    baseline: float,
    mde_range: list[float] | np.ndarray,
    n_per_variant: int,
    alpha: float = 0.05,
    ax: Any | None = None,
) -> Any:
    """Plot statistical power vs. minimum detectable effect.

    Uses a two-proportion z-test approximation when ``baseline`` is in
    (0, 1), otherwise a two-sample t-test approximation.

    Parameters
    ----------
    baseline : float
        Baseline metric value (e.g. conversion rate).
    mde_range : list[float] | ndarray
        Range of MDE values to evaluate.
    n_per_variant : int
        Sample size per variant.
    alpha : float
        Significance level.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import norm

    fig, ax = _resolve_ax(ax)

    mde_arr = np.asarray(mde_range, dtype=float)
    z_alpha = norm.ppf(1 - alpha / 2)

    powers = []
    for mde in mde_arr:
        p1 = baseline
        p2 = baseline + mde
        # Two-proportion z-test standard error
        se = np.sqrt(p1 * (1 - p1) / n_per_variant + p2 * (1 - p2) / n_per_variant)
        if se == 0:
            powers.append(1.0)
        else:
            z_power = (abs(mde) / se) - z_alpha
            powers.append(float(norm.cdf(z_power)))

    ax.plot(mde_arr, powers, color="steelblue", linewidth=1.5)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, label="80% power")

    # Vertical line at the middle MDE (target)
    target_mde = mde_arr[len(mde_arr) // 2]
    ax.axvline(
        target_mde, color="orange", linestyle="--",
        linewidth=0.8, label=f"Target MDE={target_mde:.4f}",
    )

    ax.set_xlabel("Minimum Detectable Effect")
    ax.set_ylabel("Power")
    ax.set_title("Power Curve")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Funnel chart
# ---------------------------------------------------------------------------


def funnel_chart(
    funnel_result: Any,
    ax: Any | None = None,
) -> Any:
    """Bar chart of conversion rates per funnel step for control and treatment.

    Parameters
    ----------
    funnel_result : FunnelResult
        Result containing ``step_results`` with per-step data.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _resolve_ax(ax)

    steps = funnel_result.step_results
    step_names = [s["name"] for s in steps]
    control_rates = [s["control_rate"] for s in steps]
    treatment_rates = [s["treatment_rate"] for s in steps]

    x = np.arange(len(step_names))
    width = 0.35

    ax.bar(x - width / 2, control_rates, width, label="Control", color="gray", alpha=0.8)
    ax.bar(x + width / 2, treatment_rates, width, label="Treatment", color="steelblue", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=30, ha="right")
    ax.set_ylabel("Conversion Rate")
    ax.set_title("Funnel Chart")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Metric comparison
# ---------------------------------------------------------------------------


def metric_comparison(
    results_dict: dict[str, Any],
    ax: Any | None = None,
) -> Any:
    """Side-by-side bar chart of lifts with CI error bars.

    Parameters
    ----------
    results_dict : dict[str, ExperimentResult]
        Mapping of metric name to experiment result.
    ax : matplotlib.axes.Axes | None
        Axes to draw on.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _resolve_ax(ax)

    names = list(results_dict.keys())
    lifts = [results_dict[n].lift for n in names]
    ci_lowers = [results_dict[n].ci_lower for n in names]
    ci_uppers = [results_dict[n].ci_upper for n in names]

    errors_lower = [lift - ci_lo for lift, ci_lo in zip(lifts, ci_lowers, strict=False)]
    errors_upper = [ci_hi - lift for lift, ci_hi in zip(lifts, ci_uppers, strict=False)]

    x = np.arange(len(names))
    colors = [
        "steelblue" if results_dict[n].significant else "gray" for n in names
    ]

    ax.bar(
        x,
        lifts,
        yerr=[errors_lower, errors_upper],
        capsize=4,
        color=colors,
        alpha=0.8,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Lift")
    ax.set_title("Metric Comparison")
    fig.tight_layout()
    return fig
