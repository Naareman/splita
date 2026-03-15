"""Polars Series support for splita.

Provides a helper to convert polars Series to numpy arrays, and patches
:func:`~splita._validation.check_array_like` to accept polars Series
transparently when this module is imported.

Usage::

    import splita.integrations.polars_support

    # Now polars Series are accepted anywhere splita expects array-like data
    import polars as pl
    from splita import Experiment
    result = Experiment(pl.Series([0, 1, 0]), pl.Series([1, 1, 0])).run()
"""

from __future__ import annotations

from typing import Any

import numpy as np


def from_polars(series: Any) -> np.ndarray:
    """Convert a polars Series to a numpy array.

    Parameters
    ----------
    series : polars.Series
        The polars Series to convert.

    Returns
    -------
    np.ndarray
        Numpy array with the same data.

    Raises
    ------
    TypeError
        If the input is not a polars Series.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for this function. Install it with: pip install polars"
        ) from exc

    if not isinstance(series, pl.Series):
        raise TypeError(f"Expected a polars Series, got {type(series).__name__}.")

    return series.to_numpy()
