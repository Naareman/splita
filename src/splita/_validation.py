from __future__ import annotations

import math
import warnings

import numpy as np

# ─── Internal helper ────────────────────────────────────────────────


def format_error(
    problem: str,
    detail: str | None = None,
    hint: str | None = None,
) -> str:
    """Assemble a 3-part error string (problem / detail / hint).

    Parameters
    ----------
    problem : str
        What went wrong -- use "must" or "can't".
    detail : str, optional
        The actual bad value and why it's wrong.
    hint : str, optional
        Suggested fix. Only include when confident.

    Returns
    -------
    str
        Formatted multi-line error message.
    """
    parts = [problem]
    if detail is not None:
        parts.append(f"  Detail: {detail}")
    if hint is not None:
        parts.append(f"  Hint: {hint}")
    return "\n".join(parts)


# ─── Range / numeric checks ────────────────────────────────────────


def check_in_range(
    value: float,
    name: str,
    low: float,
    high: float,
    *,
    low_inclusive: bool = False,
    high_inclusive: bool = False,
    hint: str | None = None,
) -> None:
    """Validate that *value* falls within a numeric range.

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        Parameter name, used in the error message.
    low : float
        Lower bound.
    high : float
        Upper bound.
    low_inclusive : bool, default False
        Whether the lower bound is inclusive (``[``) or exclusive (``(``).
    high_inclusive : bool, default False
        Whether the upper bound is inclusive (``]``) or exclusive (``)``).
    hint : str, optional
        Suggested fix appended to the error message.

    Raises
    ------
    ValueError
        If *value* is outside the specified range.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(
            format_error(
                f"`{name}` must be a finite number, got {value}.",
                hint="check for missing or infinite values in your input.",
            )
        )

    left = "[" if low_inclusive else "("
    right = "]" if high_inclusive else ")"
    range_str = f"{left}{low}, {high}{right}"

    too_low = value < low if low_inclusive else value <= low
    too_high = value > high if high_inclusive else value >= high

    if too_low:
        raise ValueError(
            format_error(
                f"`{name}` must be in {range_str}, got {value}.",
                f"value is below the minimum of {low}.",
                hint,
            )
        )
    if too_high:
        raise ValueError(
            format_error(
                f"`{name}` must be in {range_str}, got {value}.",
                f"value is above the maximum of {high}.",
                hint,
            )
        )


def check_positive(
    value: float,
    name: str,
    *,
    allow_zero: bool = False,
    hint: str | None = None,
) -> None:
    """Validate that *value* is positive (or non-negative).

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        Parameter name, used in the error message.
    allow_zero : bool, default False
        If True, zero is accepted.
    hint : str, optional
        Suggested fix appended to the error message.

    Raises
    ------
    ValueError
        If *value* is not positive (or not non-negative when *allow_zero* is True).
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(
            format_error(
                f"`{name}` must be a finite number, got {value}.",
                hint="check for missing or infinite values in your input.",
            )
        )

    if allow_zero:
        if value < 0:
            raise ValueError(
                format_error(
                    f"`{name}` must be >= 0, got {value}.",
                    "value is negative.",
                    hint,
                )
            )
    else:
        if value <= 0:
            raise ValueError(
                format_error(
                    f"`{name}` must be > 0, got {value}.",
                    "value must be strictly positive.",
                    hint,
                )
            )


def check_is_integer(
    value: float | int,
    name: str,
    *,
    min_value: int | None = None,
    hint: str | None = None,
) -> None:
    """Validate that *value* is an integer (or an int-like float such as 5.0).

    Parameters
    ----------
    value : float or int
        The value to validate.
    name : str
        Parameter name, used in the error message.
    min_value : int, optional
        If given, also enforce ``value >= min_value``.
    hint : str, optional
        Suggested fix appended to the error message.

    Raises
    ------
    TypeError
        If *value* is not numeric.
    ValueError
        If *value* is not integer-valued or is below *min_value*.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            format_error(
                f"`{name}` must be an integer, got type {type(value).__name__}.",
            )
        )

    # Check for NaN/inf before int check
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        raise ValueError(
            format_error(
                f"`{name}` must be a finite integer, got {value}.",
            )
        )

    if isinstance(value, float) and value != int(value):
        raise ValueError(
            format_error(
                f"`{name}` must be an integer, got {value}.",
                f"{value} is not a whole number.",
                hint,
            )
        )

    if min_value is not None and value < min_value:
        raise ValueError(
            format_error(
                f"`{name}` must be >= {min_value}, got {int(value)}.",
                hint=hint,
            )
        )


# ─── Array checks ──────────────────────────────────────────────────


def check_array_like(
    value: object,
    name: str,
    *,
    min_length: int | None = None,
    allow_nan: bool = False,
    dtype: str = "float64",
) -> np.ndarray:
    """Validate and convert an array-like input to a NumPy array.

    Parameters
    ----------
    value : array-like
        Input data (list, tuple, ndarray, or pandas Series).
    name : str
        Parameter name, used in error/warning messages.
    min_length : int, optional
        Minimum required length (checked after NaN removal).
    allow_nan : bool, default False
        If False and NaNs are found, emit a ``RuntimeWarning``, drop them,
        and return the cleaned array.
    dtype : str, default "float64"
        NumPy dtype to cast to.

    Returns
    -------
    np.ndarray
        Cleaned, typed array.

    Raises
    ------
    TypeError
        If *value* cannot be converted to an array.
    ValueError
        If the resulting array is shorter than *min_length*.
    """
    # Handle pandas Series
    try:
        import pandas as pd

        if isinstance(pd.Series, type) and isinstance(value, pd.Series):
            value = value.to_numpy()
    except ImportError:  # pragma: no cover
        pass

    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(
            format_error(
                f"`{name}` must be array-like (list, tuple, or ndarray), "
                f"got type {type(value).__name__}.",
            )
        )

    try:
        arr = np.asarray(value, dtype=dtype)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            format_error(
                f"`{name}` can't be converted to a numeric array.",
                str(exc),
            )
        ) from exc

    if arr.ndim > 1:
        raise ValueError(
            format_error(
                f"`{name}` must be a 1-D array, got {arr.ndim}-D array with shape {arr.shape}.",
                hint="pass a 1-D list, numpy array, or pandas Series.",
            )
        )

    if not allow_nan:
        nan_mask = np.isnan(arr)
        nan_count = int(nan_mask.sum())
        if nan_count > 0:
            total = len(arr)
            s = "s" if nan_count > 1 else ""
            have = "have" if nan_count > 1 else "has"
            warnings.warn(
                f"`{name}` contains {nan_count} NaN value{s} (out of {total}). "
                f"NaN value{s} {have} been dropped.",
                RuntimeWarning,
                stacklevel=2,
            )
            arr = arr[~nan_mask]

        inf_mask = np.isinf(arr)
        if np.any(inf_mask):
            n_inf = int(np.sum(inf_mask))
            warnings.warn(
                f"`{name}` contains {n_inf} infinite value{'s' if n_inf > 1 else ''} "
                f"(out of {len(arr)}). Infinite values have been dropped.",
                RuntimeWarning,
                stacklevel=2,
            )
            arr = arr[~inf_mask]

    if min_length is not None and len(arr) < min_length:
        raise ValueError(
            format_error(
                f"`{name}` must have at least {min_length} elements, "
                f"got {len(arr)} (after NaN removal).",
            )
        )

    return arr


# ─── Categorical / structural checks ───────────────────────────────


def check_one_of(
    value: str,
    name: str,
    options: list[str],
    *,
    hint: str | None = None,
) -> None:
    """Validate that *value* is one of a set of allowed options.

    Parameters
    ----------
    value : str
        The value to validate.
    name : str
        Parameter name, used in the error message.
    options : list of str
        Allowed values.
    hint : str, optional
        If not given, a closest-match suggestion is attempted automatically.

    Raises
    ------
    ValueError
        If *value* is not in *options*.
    """
    if value in options:
        return

    if hint is None:
        hint = _suggest_match(value, options)

    raise ValueError(
        format_error(
            f"`{name}` must be one of {tuple(options)}, got {value!r}.",
            hint=hint,
        )
    )


def _suggest_match(value: str, options: list[str]) -> str | None:
    """Return a 'did you mean ...?' hint using simple substring matching."""
    value_lower = value.lower()
    for opt in options:
        opt_lower = opt.lower()
        if value_lower in opt_lower or opt_lower in value_lower:
            return f"did you mean {opt!r}?"
    return None


def check_same_length(
    arr1: np.ndarray,
    arr2: np.ndarray,
    name1: str,
    name2: str,
) -> None:
    """Validate that two arrays have the same length.

    Parameters
    ----------
    arr1 : np.ndarray
        First array.
    arr2 : np.ndarray
        Second array.
    name1 : str
        Name for the first array.
    name2 : str
        Name for the second array.

    Raises
    ------
    ValueError
        If the arrays differ in length.
    """
    if len(arr1) != len(arr2):
        raise ValueError(
            format_error(
                f"`{name1}` and `{name2}` must have the same length.",
                f"{name1} has {len(arr1)} elements, {name2} has {len(arr2)} elements.",
            )
        )


def check_not_empty(value: object, name: str) -> None:
    """Validate that a sequence or array is not empty.

    Parameters
    ----------
    value : sized
        The sequence or array to check.
    name : str
        Parameter name, used in the error message.

    Raises
    ------
    ValueError
        If *value* has length 0.
    """
    if len(value) == 0:  # type: ignore[arg-type]
        raise ValueError(
            format_error(
                f"`{name}` can't be empty.",
                "received a sequence with 0 elements.",
            )
        )


def check_probabilities_sum_to_one(
    values: object,
    name: str,
    *,
    tol: float = 1e-6,
) -> None:
    """Validate that a sequence of probabilities sums to 1.0 within tolerance.

    Parameters
    ----------
    values : array-like
        Sequence of probability values.
    name : str
        Parameter name, used in the error message.
    tol : float, default 1e-6
        Absolute tolerance for the sum check.

    Raises
    ------
    ValueError
        If the sum deviates from 1.0 by more than *tol*.
    """
    total = float(np.sum(values))  # type: ignore[arg-type]
    if abs(total - 1.0) > tol:
        raise ValueError(
            format_error(
                f"`{name}` must sum to 1.0, got {total}.",
                f"the sum deviates from 1.0 by {abs(total - 1.0):.2e}.",
                "ensure all probabilities are normalized.",
            )
        )
