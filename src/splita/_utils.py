from __future__ import annotations

import math

import numpy as np

from splita._validation import format_error


# ─── RNG handling ─────────────────────────────────────────────────


def ensure_rng(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    """Convert a random_state argument to a NumPy Generator.

    Follows the NumPy / scikit-learn convention for seed handling.

    Parameters
    ----------
    random_state : int, np.random.Generator, or None
        - ``None`` → ``np.random.default_rng()`` (OS entropy).
        - ``int`` → ``np.random.default_rng(seed)``.
        - ``np.random.Generator`` → passed through unchanged.

    Returns
    -------
    np.random.Generator
        A NumPy random Generator instance.

    Raises
    ------
    TypeError
        If *random_state* is not one of the accepted types.

    Examples
    --------
    >>> rng = ensure_rng(42)
    >>> isinstance(rng, np.random.Generator)
    True
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise TypeError(
        format_error(
            f"`random_state` must be None, an int, or a np.random.Generator, "
            f"got {type(random_state).__name__}.",
            detail=f"received {type(random_state)}, which is not a supported seed type.",
            hint="pass an integer seed, a Generator, or None for OS entropy.",
        )
    )


# ─── Array conversion ────────────────────────────────────────────


def to_array(data: object, name: str, dtype: str = "float64") -> np.ndarray:
    """Convert array-like input to a 1-D NumPy array.

    A thin conversion wrapper. Does **not** handle NaN removal — use
    :func:`splita._validation.check_array_like` for that.

    Parameters
    ----------
    data : array-like
        Input data (list, tuple, ndarray, or pandas Series).
    name : str
        Parameter name, used in error messages.
    dtype : str, default "float64"
        NumPy dtype string to cast to.

    Returns
    -------
    np.ndarray
        1-D array of the requested dtype.

    Raises
    ------
    TypeError
        If *data* cannot be converted to the requested dtype.

    Examples
    --------
    >>> to_array([1, 2, 3], "values")
    array([1., 2., 3.])
    """
    # Handle pandas Series
    try:
        import pandas as pd

        if isinstance(data, pd.Series):
            data = data.to_numpy()
    except ImportError:
        pass

    try:
        arr = np.asarray(data, dtype=dtype)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            format_error(
                f"`{name}` can't be converted to {dtype}.",
                detail=f"received {type(data)}, which is not numeric.",
                hint="pass a list, numpy array, or pandas Series of numbers.",
            )
        ) from exc

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        raise ValueError(
            format_error(
                f"`{name}` must be a 1-D array, got {arr.ndim}-D array with shape {arr.shape}.",
                hint="pass a 1-D list, numpy array, or pandas Series.",
            )
        )

    return arr


# ─── Effect-size measures ─────────────────────────────────────────


def cohens_d(control: np.ndarray, treatment: np.ndarray) -> float:
    """Compute Cohen's d (standardised mean difference) for continuous metrics.

    Uses the pooled standard deviation as the denominator.

    Parameters
    ----------
    control : np.ndarray
        Observations from the control group.
    treatment : np.ndarray
        Observations from the treatment group.

    Returns
    -------
    float
        Cohen's d.  Positive values indicate the treatment mean exceeds the
        control mean.

    Examples
    --------
    >>> cohens_d(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
    1.0
    """
    n1, n2 = len(control), len(treatment)
    mean_diff = float(np.mean(treatment) - np.mean(control))
    denom = n1 + n2 - 2
    if denom <= 0:
        pooled_std = 0.0
    else:
        s1, s2 = float(np.std(control, ddof=1)), float(np.std(treatment, ddof=1))
        pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / denom)
    if pooled_std == 0.0:
        if mean_diff == 0.0:
            return 0.0
        import warnings

        warnings.warn(
            "Pooled standard deviation is zero — Cohen's d is undefined. "
            "Returning inf.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf") if mean_diff > 0 else float("-inf")
    return mean_diff / pooled_std


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h for proportion metrics.

    Parameters
    ----------
    p1 : float
        First proportion (e.g., control conversion rate).
    p2 : float
        Second proportion (e.g., treatment conversion rate).

    Returns
    -------
    float
        Cohen's h.  Positive values indicate ``p2 > p1``.

    Examples
    --------
    >>> round(cohens_h(0.5, 0.7), 4)
    0.4115
    """
    for val, pname in [(p1, "p1"), (p2, "p2")]:
        if not 0 <= val <= 1:
            raise ValueError(
                format_error(
                    f"`{pname}` must be in [0, 1], got {val}.",
                    hint="proportions must be between 0 and 1 inclusive.",
                )
            )
    return float(2.0 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1))))


# ─── Lift & proportions ──────────────────────────────────────────


def relative_lift(control_mean: float, treatment_mean: float) -> float:
    """Compute relative lift: ``(treatment - control) / |control|``.

    Parameters
    ----------
    control_mean : float
        Mean of the control group.
    treatment_mean : float
        Mean of the treatment group.

    Returns
    -------
    float
        Relative lift.  Returns ``inf`` / ``-inf`` when *control_mean* is
        zero and *treatment_mean* is nonzero, or ``0.0`` when both are zero.

    Examples
    --------
    >>> relative_lift(100.0, 120.0)
    0.2
    """
    if control_mean == 0.0:
        if treatment_mean > 0.0:
            return float("inf")
        if treatment_mean < 0.0:
            return float("-inf")
        return 0.0
    return (treatment_mean - control_mean) / abs(control_mean)


def pooled_proportion(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute pooled proportion from two binary arrays.

    Parameters
    ----------
    x1 : np.ndarray
        Binary array from group 1.
    x2 : np.ndarray
        Binary array from group 2.

    Returns
    -------
    float
        ``(sum(x1) + sum(x2)) / (len(x1) + len(x2))``.

    Examples
    --------
    >>> pooled_proportion(np.array([1, 0, 1]), np.array([0, 0, 1]))
    0.5
    """
    total = len(x1) + len(x2)
    if total == 0:
        raise ValueError(
            format_error(
                "`x1` and `x2` must not both be empty.",
                hint="pass non-empty binary arrays.",
            )
        )
    return float((np.sum(x1) + np.sum(x2)) / total)


# ─── Metric detection ────────────────────────────────────────────


def auto_detect_metric(data: np.ndarray) -> str:
    """Detect metric type from data values.

    Parameters
    ----------
    data : np.ndarray
        Observations to inspect.

    Returns
    -------
    str
        ``"conversion"`` if every value is 0 or 1, ``"continuous"`` otherwise.

    Notes
    -----
    Auto-detection is based on unique values only. Data that happens to
    contain only 0s and 1s (e.g., a low-count metric) will be classified
    as ``"conversion"``. When this is ambiguous, specify the metric type
    explicitly via ``metric="continuous"``.

    Examples
    --------
    >>> auto_detect_metric(np.array([0, 1, 1, 0, 1]))
    'conversion'
    >>> auto_detect_metric(np.array([1.5, 2.3, 0.7]))
    'continuous'
    """
    if len(data) == 0:
        raise ValueError(
            format_error(
                "`data` must not be empty for metric detection.",
                hint="pass a non-empty array of observations.",
            )
        )
    unique = np.unique(data)
    if np.all((unique == 0) | (unique == 1)):
        return "conversion"
    return "continuous"
