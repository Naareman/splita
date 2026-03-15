"""FlickerDetector — detect users who switched variants mid-experiment.

Flicker (a.k.a. variant instability) happens when a user is assigned
to variant A, then later assigned to variant B. This contaminates
intent-to-treat estimates.
"""

from __future__ import annotations

import numpy as np

from splita._types import FlickerResult
from splita._validation import (
    check_in_range,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class FlickerDetector:
    """Detect users who switched between variants during an experiment.

    A "flicker" is a user who appears with more than one unique variant
    assignment. High flicker rates indicate randomisation instability
    and can bias experiment results.

    Parameters
    ----------
    threshold : float, default 0.01
        Flicker rate above which the result is flagged as problematic.

    Examples
    --------
    >>> import numpy as np
    >>> user_ids = np.array([1, 2, 3, 1, 2, 3])
    >>> variants = np.array([0, 1, 0, 0, 1, 0])
    >>> result = FlickerDetector().detect(user_ids, variants)
    >>> result.flicker_rate
    0.0
    """

    def __init__(self, *, threshold: float = 0.01):
        check_in_range(
            threshold,
            "threshold",
            0.0,
            1.0,
            low_inclusive=True,
            high_inclusive=True,
            hint="threshold is a fraction between 0 and 1.",
        )
        self._threshold = threshold

    def detect(
        self,
        user_ids: ArrayLike,
        variant_assignments: ArrayLike,
        timestamps: ArrayLike | None = None,
    ) -> FlickerResult:
        """Detect flicker in variant assignments.

        Parameters
        ----------
        user_ids : array-like
            User identifiers. Each element corresponds to one observation.
        variant_assignments : array-like
            Variant assignment for each observation (e.g. 0 or 1).
        timestamps : array-like or None, default None
            Optional timestamps. Currently used for validation only;
            future versions may use temporal ordering.

        Returns
        -------
        FlickerResult
            Detection result including flicker rate and affected users.

        Raises
        ------
        TypeError
            If inputs cannot be converted to arrays.
        ValueError
            If inputs are empty or have mismatched lengths.
        """
        # Validate — use object dtype for user_ids to preserve IDs
        if not isinstance(user_ids, (list, tuple, np.ndarray)):
            raise TypeError(
                format_error(
                    "`user_ids` must be array-like (list, tuple, or ndarray), "
                    f"got type {type(user_ids).__name__}.",
                )
            )

        uid_arr = np.asarray(user_ids)
        if uid_arr.ndim != 1:
            raise ValueError(
                format_error(
                    f"`user_ids` must be a 1-D array, got {uid_arr.ndim}-D.",
                )
            )

        if len(uid_arr) == 0:
            raise ValueError(
                format_error(
                    "`user_ids` can't be empty.",
                    "received a sequence with 0 elements.",
                )
            )

        # Validate variant_assignments
        if not isinstance(variant_assignments, (list, tuple, np.ndarray)):
            raise TypeError(
                format_error(
                    "`variant_assignments` must be array-like (list, tuple, or ndarray), "
                    f"got type {type(variant_assignments).__name__}.",
                )
            )

        var_arr = np.asarray(variant_assignments)
        if var_arr.ndim != 1:
            raise ValueError(
                format_error(
                    f"`variant_assignments` must be a 1-D array, got {var_arr.ndim}-D.",
                )
            )

        if len(uid_arr) != len(var_arr):
            raise ValueError(
                format_error(
                    "`user_ids` and `variant_assignments` must have the same length.",
                    f"user_ids has {len(uid_arr)} elements, "
                    f"variant_assignments has {len(var_arr)} elements.",
                )
            )

        # Validate timestamps if provided
        if timestamps is not None:
            ts_arr = np.asarray(timestamps)
            if ts_arr.ndim != 1:
                raise ValueError(
                    format_error(
                        f"`timestamps` must be a 1-D array, got {ts_arr.ndim}-D.",
                    )
                )
            if len(ts_arr) != len(uid_arr):
                raise ValueError(
                    format_error(
                        "`timestamps` and `user_ids` must have the same length.",
                        f"timestamps has {len(ts_arr)} elements, "
                        f"user_ids has {len(uid_arr)} elements.",
                    )
                )

        # Find flickers: users with >1 unique variant
        unique_users = np.unique(uid_arr)
        n_users = len(unique_users)
        flicker_users_list = []

        for uid in unique_users:
            mask = uid_arr == uid
            variants_for_user = var_arr[mask]
            if len(np.unique(variants_for_user)) > 1:
                flicker_users_list.append(uid)

        n_flickers = len(flicker_users_list)
        flicker_rate = n_flickers / n_users if n_users > 0 else 0.0
        is_problematic = flicker_rate > self._threshold

        if n_flickers == 0:
            message = f"No flickers detected among {n_users} users."
        else:
            pct = flicker_rate * 100
            message = f"{n_flickers} of {n_users} users ({pct:.1f}%) flickered between variants."
            if is_problematic:
                message += " Flicker rate exceeds threshold; results may be biased."

        # Convert numpy types in flicker_users_list for JSON serialisability
        flicker_users_out = [
            int(u) if isinstance(u, (np.integer,)) else u for u in flicker_users_list
        ]

        return FlickerResult(
            flicker_rate=float(flicker_rate),
            n_flickers=n_flickers,
            n_users=n_users,
            is_problematic=is_problematic,
            flicker_users=flicker_users_out,
            message=message,
        )
