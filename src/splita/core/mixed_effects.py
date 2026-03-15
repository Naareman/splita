"""Mixed-effects experiment — repeated measures per user.

Estimates the average treatment effect accounting for within-user
correlation using a simplified random-intercept model. This avoids
the need for full mixed-effects model fitting (Bates et al., lme4)
by using within-user centering and between-user adjustment.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from splita._types import MixedEffectsResult
from splita._validation import (
    check_array_like,
    check_in_range,
    check_same_length,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class MixedEffectsExperiment:
    """Mixed-effects analysis for repeated-measures A/B tests.

    Estimates the treatment effect with a random intercept per user,
    properly accounting for within-user correlation in experiments
    where each user contributes multiple observations (e.g. sessions,
    purchases, page views).

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for inference.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n_users = 100
    >>> obs_per_user = 5
    >>> user_ids = np.repeat(np.arange(n_users), obs_per_user)
    >>> treatment = np.repeat(rng.binomial(1, 0.5, n_users), obs_per_user).astype(float)
    >>> user_effects = rng.normal(0, 1, n_users)
    >>> outcome = (
    ...     np.repeat(user_effects, obs_per_user)
    ...     + 0.5 * treatment
    ...     + rng.normal(0, 1, n_users * obs_per_user)
    ... )
    >>> me = MixedEffectsExperiment()
    >>> me.fit(outcome, treatment, user_ids)  # doctest: +ELLIPSIS
    <splita.core.mixed_effects.MixedEffectsExperiment object at ...>
    >>> r = me.result()
    >>> r.n_users == 100
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._alpha = alpha
        self._result: MixedEffectsResult | None = None

    def fit(
        self,
        outcome: ArrayLike,
        treatment: ArrayLike,
        user_ids: ArrayLike,
    ) -> MixedEffectsExperiment:
        """Fit the mixed-effects model.

        Uses a two-step approach:

        1. Compute user-level means of outcome and treatment.
        2. Regress user-level mean outcome on user-level mean treatment,
           with standard errors adjusted for within-user clustering.

        Parameters
        ----------
        outcome : array-like
            Outcome variable for each observation.
        treatment : array-like
            Treatment indicator for each observation (must be constant
            within each user).
        user_ids : array-like
            User identifier for each observation.

        Returns
        -------
        MixedEffectsExperiment
            The fitted estimator (self).

        Raises
        ------
        TypeError
            If inputs cannot be converted to numeric arrays.
        ValueError
            If arrays have fewer than 4 elements, mismatched lengths,
            or fewer than 2 unique users.
        """
        y = check_array_like(outcome, "outcome", min_length=4)
        t = check_array_like(treatment, "treatment", min_length=4)
        u = check_array_like(user_ids, "user_ids", min_length=4)

        check_same_length(y, t, "outcome", "treatment")
        check_same_length(y, u, "outcome", "user_ids")

        n_obs = len(y)
        unique_users = np.unique(u)
        n_users = len(unique_users)

        if n_users < 2:
            raise ValueError(
                format_error(
                    "`user_ids` must contain at least 2 unique users.",
                    f"found {n_users} unique user(s).",
                    "ensure treatment and control each have at least 1 user.",
                )
            )

        # Compute user-level aggregates
        user_means_y = np.empty(n_users)
        user_means_t = np.empty(n_users)
        user_counts = np.empty(n_users)

        for i, uid in enumerate(unique_users):
            mask = u == uid
            user_means_y[i] = np.mean(y[mask])
            user_means_t[i] = np.mean(t[mask])
            user_counts[i] = np.sum(mask)

        # Check that we have both treatment and control users
        ctrl_mask = user_means_t < 0.5
        trt_mask = user_means_t >= 0.5

        if not np.any(ctrl_mask) or not np.any(trt_mask):
            raise ValueError(
                format_error(
                    "Both treatment and control groups must have at least 1 user.",
                    f"found {int(np.sum(trt_mask))} treatment users and "
                    f"{int(np.sum(ctrl_mask))} control users.",
                )
            )

        # ICC computation
        # Between-user variance and within-user variance
        grand_mean = np.mean(y)
        ss_between = 0.0
        ss_within = 0.0

        for _i, uid in enumerate(unique_users):
            mask = u == uid
            user_y = y[mask]
            n_i = len(user_y)
            user_mean = np.mean(user_y)
            ss_between += n_i * (user_mean - grand_mean) ** 2
            ss_within += np.sum((user_y - user_mean) ** 2)

        df_between = n_users - 1
        df_within = n_obs - n_users

        ms_between = ss_between / max(df_between, 1)
        ms_within = ss_within / max(df_within, 1)

        avg_n = n_obs / n_users
        # ICC from one-way ANOVA
        if ms_between > ms_within:
            sigma2_between = (ms_between - ms_within) / avg_n
            icc = sigma2_between / (sigma2_between + ms_within)
        else:
            icc = 0.0

        # Treatment effect: user-level regression
        # Simple difference of means at user level
        mean_y_trt = np.mean(user_means_y[trt_mask])
        mean_y_ctrl = np.mean(user_means_y[ctrl_mask])
        ate = mean_y_trt - mean_y_ctrl

        # SE with clustering adjustment
        n_trt = int(np.sum(trt_mask))
        n_ctrl = int(np.sum(ctrl_mask))

        var_trt = float(np.var(user_means_y[trt_mask], ddof=1)) if n_trt > 1 else 0.0
        var_ctrl = float(np.var(user_means_y[ctrl_mask], ddof=1)) if n_ctrl > 1 else 0.0

        se = float(np.sqrt(var_trt / n_trt + var_ctrl / n_ctrl))

        # Inference
        if se > 0:
            z = ate / se
            pvalue = float(2 * norm.sf(abs(z)))
        else:
            pvalue = 1.0 if ate == 0 else 0.0  # pragma: no cover

        z_crit = float(norm.ppf(1 - self._alpha / 2))
        ci_lower = ate - z_crit * se
        ci_upper = ate + z_crit * se

        self._result = MixedEffectsResult(
            ate=float(ate),
            se=se,
            pvalue=pvalue,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=pvalue < self._alpha,
            icc=float(icc),
            n_users=n_users,
            n_observations=n_obs,
            alpha=self._alpha,
        )
        return self

    def result(self) -> MixedEffectsResult:
        """Return the mixed-effects result.

        Returns
        -------
        MixedEffectsResult
            The estimated treatment effect and diagnostics.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._result is None:
            raise RuntimeError(
                format_error(
                    "MixedEffectsExperiment must be fitted before calling result().",
                    "call .fit() with outcome, treatment, and user_ids first.",
                )
            )
        return self._result
