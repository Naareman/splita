"""Fractional factorial design — generate and analyse multi-factor experiments.

Implements resolution III and higher fractional factorial designs
(Box & Hunter 1961) for efficiently screening many factors with
fewer experimental runs than a full 2^N factorial. Supports the
MOST framework (Collins 2018) for building optimised interventions.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import norm

from splita._types import FactorialResult
from splita._validation import (
    check_array_like,
    check_is_integer,
    format_error,
)

ArrayLike = list | tuple | np.ndarray


class FractionalFactorialDesign:
    """Generate and analyse fractional factorial designs.

    Produces design matrices for testing N factors with fewer than
    2^N experimental cells, and analyses outcomes to identify
    significant main effects and interactions.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for identifying significant effects.

    Examples
    --------
    >>> ffd = FractionalFactorialDesign()
    >>> matrix = ffd.generate(4, resolution=3)
    >>> matrix.shape[1] == 4
    True
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha represents the false positive rate.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )
        self._alpha = alpha

    def generate(
        self,
        n_factors: int,
        resolution: int = 3,
    ) -> np.ndarray:
        """Generate a fractional factorial design matrix.

        Creates a 2^(k-p) fractional factorial design where k is
        ``n_factors`` and p is chosen to achieve the requested
        resolution.

        Parameters
        ----------
        n_factors : int
            Number of factors (must be >= 2).
        resolution : int, default 3
            Minimum resolution of the design (3, 4, or 5).
            Resolution III: main effects not aliased with each other.
            Resolution IV: main effects not aliased with 2FI.
            Resolution V: 2FI not aliased with each other.

        Returns
        -------
        np.ndarray
            Design matrix of shape (n_runs, n_factors) with values
            in {-1, +1}.

        Raises
        ------
        ValueError
            If ``n_factors < 2`` or ``resolution`` is not in {3, 4, 5}.
        """
        check_is_integer(n_factors, "n_factors", min_value=2)
        check_is_integer(resolution, "resolution", min_value=3)

        if resolution > 5:
            raise ValueError(
                format_error(
                    f"`resolution` must be 3, 4, or 5, got {resolution}.",
                    "higher resolutions are not supported.",
                )
            )

        n_factors = int(n_factors)
        resolution = int(resolution)

        # Determine number of base factors needed
        # For resolution III: k_base = ceil(log2(n_factors + 1))
        # For resolution IV/V: need more base factors
        if resolution == 3:
            k_base = max(2, int(np.ceil(np.log2(n_factors + 1))))
        elif resolution == 4:
            k_base = max(2, int(np.ceil(np.log2(n_factors))) + 1)
        else:  # resolution 5
            k_base = max(3, n_factors - 1)

        # Ensure k_base <= n_factors (for small n_factors, full factorial)
        k_base = min(k_base, n_factors)

        n_runs = 2**k_base

        # Generate full factorial for base factors
        base_matrix = np.zeros((n_runs, k_base), dtype=float)
        for col in range(k_base):
            period = 2 ** (k_base - col - 1)
            for row in range(n_runs):
                base_matrix[row, col] = 1.0 if (row // period) % 2 == 0 else -1.0

        if n_factors <= k_base:
            return base_matrix[:, :n_factors]

        # Generate additional columns as products of base columns
        design = np.zeros((n_runs, n_factors), dtype=float)
        design[:, :k_base] = base_matrix

        # Create additional columns from interactions of base columns
        col_idx = k_base
        # Start with higher-order interactions for better resolution
        for order in range(2, k_base + 1):
            if col_idx >= n_factors:
                break
            for combo in combinations(range(k_base), order):
                if col_idx >= n_factors:
                    break
                col = np.ones(n_runs, dtype=float)
                for c in combo:
                    col *= base_matrix[:, c]
                design[:, col_idx] = col
                col_idx += 1

        return design

    def analyze(
        self,
        outcomes: ArrayLike,
        design_matrix: np.ndarray,
        factor_names: list[str] | None = None,
    ) -> FactorialResult:
        """Analyse outcomes from a factorial experiment.

        Estimates main effects and two-factor interactions using
        half the difference in mean outcomes between +1 and -1
        levels of each factor.

        Parameters
        ----------
        outcomes : array-like
            Observed outcomes, one per experimental run.
        design_matrix : np.ndarray
            Design matrix of shape (n_runs, n_factors) with values
            in {-1, +1}.
        factor_names : list of str or None, default None
            Names for each factor. If None, factors are named
            ``X1``, ``X2``, etc.

        Returns
        -------
        FactorialResult
            Main effects, interactions, and significance information.

        Raises
        ------
        ValueError
            If ``outcomes`` length does not match the design matrix rows.
        """
        y = check_array_like(outcomes, "outcomes", min_length=2)

        if not isinstance(design_matrix, np.ndarray):
            design_matrix = np.asarray(design_matrix, dtype=float)

        if design_matrix.ndim != 2:
            raise ValueError(
                format_error(
                    "`design_matrix` must be 2-D.",
                    f"got {design_matrix.ndim}-D array.",
                )
            )

        n_runs, n_factors = design_matrix.shape

        if len(y) != n_runs:
            raise ValueError(
                format_error(
                    "`outcomes` must have the same length as design_matrix rows.",
                    f"outcomes has {len(y)} elements, design_matrix has {n_runs} rows.",
                )
            )

        if factor_names is None:
            factor_names = [f"X{i + 1}" for i in range(n_factors)]
        elif len(factor_names) != n_factors:
            raise ValueError(
                format_error(
                    "`factor_names` must have one name per factor.",
                    f"expected {n_factors} names, got {len(factor_names)}.",
                )
            )

        # Estimate main effects: effect = mean(y at +1) - mean(y at -1)
        y_std = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0

        # Compute residual SE after fitting all main effects via OLS
        # In saturated designs (n_runs <= n_factors + 1), we cannot
        # estimate residual variance so we use pool-of-effects method:
        # use the smallest |effects| as noise estimate (Lenth 1989).
        saturated = n_runs <= n_factors + 1
        X_design = np.column_stack([np.ones(n_runs), design_matrix])
        if not saturated:
            try:
                beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
                residuals = y - X_design @ beta
                df_resid = n_runs - n_factors - 1
                resid_se = float(np.sqrt(np.sum(residuals**2) / df_resid))
            except np.linalg.LinAlgError:
                resid_se = y_std
        else:
            resid_se = None  # Will use Lenth's method below

        main_effects: dict[str, float] = {}
        effect_sizes: dict[str, float] = {}
        significant_factors: list[str] = []

        # First pass: compute all effects
        effect_list: list[float] = []
        for j in range(n_factors):
            col = design_matrix[:, j]
            high = y[col > 0]
            low = y[col < 0]

            effect = float(np.mean(high) - np.mean(low)) if len(high) > 0 and len(low) > 0 else 0.0

            main_effects[factor_names[j]] = effect
            effect_list.append(effect)

            # Standardised effect size
            if y_std > 0:
                effect_sizes[factor_names[j]] = effect / y_std
            else:
                effect_sizes[factor_names[j]] = 0.0

        # For saturated designs, use Lenth's pseudo standard error (PSE)
        if saturated and len(effect_list) >= 2:
            abs_effects = np.array([abs(e) for e in effect_list])
            # s0 = 1.5 * median(|effects|)
            s0 = 1.5 * float(np.median(abs_effects))
            # PSE = 1.5 * median of |effects| <= 2.5 * s0
            trimmed = abs_effects[abs_effects <= 2.5 * s0] if s0 > 0 else abs_effects
            resid_se = 1.5 * float(np.median(trimmed)) if len(trimmed) > 0 else s0
            if resid_se == 0:
                resid_se = y_std

        # Second pass: determine significance
        for j in range(n_factors):
            effect = effect_list[j]
            col = design_matrix[:, j]
            n_high = max(int(np.sum(col > 0)), 1)
            n_low = max(int(np.sum(col < 0)), 1)
            se_contrast = resid_se * float(np.sqrt(1.0 / n_high + 1.0 / n_low))
            if se_contrast > 0:
                z = abs(effect) / se_contrast
                p = float(2 * norm.sf(z))
                if p < self._alpha:
                    significant_factors.append(factor_names[j])

        # Two-factor interactions
        interactions: dict[str, float] = {}
        for i, j in combinations(range(n_factors), 2):
            interaction_col = design_matrix[:, i] * design_matrix[:, j]
            high = y[interaction_col > 0]
            low = y[interaction_col < 0]

            effect = float(np.mean(high) - np.mean(low)) if len(high) > 0 and len(low) > 0 else 0.0

            key = f"{factor_names[i]}:{factor_names[j]}"
            interactions[key] = effect

        # Determine resolution from the design
        resolution = self._compute_resolution(design_matrix)

        return FactorialResult(
            main_effects=main_effects,
            interactions=interactions,
            significant_factors=significant_factors,
            effect_sizes=effect_sizes,
            n_runs=n_runs,
            n_factors=n_factors,
            resolution=resolution,
        )

    @staticmethod
    def _compute_resolution(design_matrix: np.ndarray) -> int:
        """Estimate the resolution of a design matrix.

        Parameters
        ----------
        design_matrix : np.ndarray
            Design matrix with values in {-1, +1}.

        Returns
        -------
        int
            Estimated resolution (3, 4, or 5).
        """
        _n_runs, n_factors = design_matrix.shape

        if n_factors <= 1:
            return 5

        # Check if any main effect is aliased with a 2FI (resolution < IV)
        for i in range(n_factors):
            for j, k in combinations(range(n_factors), 2):
                if i in (j, k):
                    continue
                interaction = design_matrix[:, j] * design_matrix[:, k]
                corr = abs(float(np.corrcoef(design_matrix[:, i], interaction)[0, 1]))
                if corr > 0.99:
                    return 3

        # Check if any 2FI is aliased with another 2FI (resolution < V)
        pairs = list(combinations(range(n_factors), 2))
        for idx1 in range(len(pairs)):
            for idx2 in range(idx1 + 1, len(pairs)):
                i1, j1 = pairs[idx1]
                i2, j2 = pairs[idx2]
                if {i1, j1} == {i2, j2}:
                    continue
                int1 = design_matrix[:, i1] * design_matrix[:, j1]
                int2 = design_matrix[:, i2] * design_matrix[:, j2]
                corr = abs(float(np.corrcoef(int1, int2)[0, 1]))
                if corr > 0.99:
                    return 4

        return 5
