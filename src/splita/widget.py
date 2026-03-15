"""Interactive Jupyter widget for sample size planning.

Provides a slider-based UI for exploring how baseline rate, MDE,
power, alpha, and daily traffic affect required sample sizes.

Requires ``ipywidgets >= 8.0``.  Install with::

    pip install splita[widget]

Examples
--------
>>> # In a Jupyter notebook:
>>> from splita.widget import sample_size_widget
>>> sample_size_widget()  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any


def sample_size_widget() -> Any:
    """Interactive ipywidgets for sample size planning in Jupyter.

    Creates sliders for baseline rate, minimum detectable effect (MDE),
    statistical power, significance level (alpha), and daily users.
    Updates the sample size calculation in real-time as sliders are
    adjusted.

    Returns
    -------
    ipywidgets.VBox
        Widget container suitable for display in a Jupyter notebook.

    Raises
    ------
    ImportError
        If ``ipywidgets`` is not installed.

    Examples
    --------
    >>> from splita.widget import sample_size_widget
    >>> w = sample_size_widget()  # doctest: +SKIP
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display  # noqa: F401
    except ImportError:
        raise ImportError(
            "ipywidgets is required for the sample size widget.\n"
            "  Hint: install with `pip install splita[widget]`."
        ) from None

    from splita.core.sample_size import SampleSize

    baseline_slider = widgets.FloatSlider(
        value=0.10,
        min=0.01,
        max=0.50,
        step=0.01,
        description="Baseline:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="400px"),
        readout_format=".2%",
    )
    mde_slider = widgets.FloatSlider(
        value=0.02,
        min=0.001,
        max=0.10,
        step=0.001,
        description="MDE:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="400px"),
        readout_format=".3f",
    )
    power_slider = widgets.FloatSlider(
        value=0.80,
        min=0.50,
        max=0.99,
        step=0.01,
        description="Power:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="400px"),
        readout_format=".0%",
    )
    alpha_slider = widgets.FloatSlider(
        value=0.05,
        min=0.01,
        max=0.20,
        step=0.01,
        description="Alpha:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="400px"),
        readout_format=".2f",
    )
    daily_slider = widgets.IntSlider(
        value=1000,
        min=100,
        max=100000,
        step=100,
        description="Daily users:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="400px"),
    )

    output = widgets.Output()

    def _update(*_args: Any) -> None:
        output.clear_output(wait=True)
        with output:
            try:
                plan = SampleSize.for_proportion(
                    baseline=baseline_slider.value,
                    mde=mde_slider.value,
                    power=power_slider.value,
                    alpha=alpha_slider.value,
                )
                plan = plan.duration(daily_slider.value)
                print(f"Sample Size per Variant: {plan.n_per_variant:,}")
                print(f"Total Sample Size:       {plan.n_total:,}")
                print(f"Estimated Days:          {plan.days_needed}")
                print(f"Relative MDE:            {plan.relative_mde:.2%}")
            except Exception as e:
                print(f"Error: {e}")

    baseline_slider.observe(_update, names="value")
    mde_slider.observe(_update, names="value")
    power_slider.observe(_update, names="value")
    alpha_slider.observe(_update, names="value")
    daily_slider.observe(_update, names="value")

    # Trigger initial calculation
    _update()

    title = widgets.HTML(
        value="<h3>Sample Size Planner</h3>"
        "<p style='color: #666; font-size: 13px;'>"
        "Adjust sliders to explore required sample sizes.</p>"
    )

    return widgets.VBox(
        [
            title,
            baseline_slider,
            mde_slider,
            power_slider,
            alpha_slider,
            daily_slider,
            output,
        ]
    )
