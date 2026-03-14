"""MultipleCorrection — Multiple testing correction for A/B/N experiments.

Adjusts p-values from multiple hypothesis tests to control the
familywise error rate (FWER) or the false discovery rate (FDR).
Supports Bonferroni, Holm, Benjamini-Hochberg, and Benjamini-Yekutieli
correction methods.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from splita._types import CorrectionResult
from splita._validation import (
    check_in_range,
    check_not_empty,
    check_one_of,
    format_error,
)

_VALID_METHODS = ["bh", "bonferroni", "holm", "by"]

_METHOD_LABELS = {
    "bh": "Benjamini-Hochberg",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "by": "Benjamini-Yekutieli",
}


class MultipleCorrection:
    """Adjust p-values for multiple comparisons.

    Corrects raw p-values from individual hypothesis tests so that the
    overall error rate is controlled at the desired ``alpha`` level.

    Parameters
    ----------
    pvalues : Sequence[float]
        Raw p-values from individual hypothesis tests.
    method : {'bh', 'bonferroni', 'holm', 'by'}, default 'bh'
        Correction method:

        - ``'bh'`` — Benjamini-Hochberg (FDR control, default).
        - ``'bonferroni'`` — Bonferroni (FWER, most conservative).
        - ``'holm'`` — Holm step-down (FWER, uniformly more powerful
          than Bonferroni).
        - ``'by'`` — Benjamini-Yekutieli (FDR under arbitrary
          dependence).
    alpha : float, default 0.05
        Familywise significance level after correction.
    labels : Sequence[str] or None, default None
        Optional human-readable names for each test.

    Examples
    --------
    >>> from splita.core.correction import MultipleCorrection
    >>> result = MultipleCorrection([0.01, 0.04, 0.20]).run()
    >>> result.n_rejected
    2
    """

    def __init__(
        self,
        pvalues: Sequence[float],
        *,
        method: Literal["bh", "bonferroni", "holm", "by"] = "bh",
        alpha: float = 0.05,
        labels: Sequence[str] | None = None,
    ) -> None:
        # --- validate pvalues ---------------------------------------------
        check_not_empty(pvalues, "pvalues")

        pv = np.asarray(pvalues, dtype=np.float64)
        if pv.ndim != 1:
            raise ValueError(
                format_error(
                    f"`pvalues` must be a 1-D sequence, got {pv.ndim}-D.",
                    hint="pass a flat list or 1-D array of p-values.",
                )
            )

        if np.any(pv < 0.0):
            bad = float(pv[pv < 0.0][0])
            raise ValueError(
                format_error(
                    f"`pvalues` must all be in [0, 1], got {bad}.",
                    "p-values cannot be negative.",
                    "check for sign errors in your test statistics.",
                )
            )

        if np.any(pv > 1.0):
            bad = float(pv[pv > 1.0][0])
            raise ValueError(
                format_error(
                    f"`pvalues` must all be in [0, 1], got {bad}.",
                    "p-values cannot exceed 1.",
                    "check that you are passing p-values, not test statistics.",
                )
            )

        # --- validate method ----------------------------------------------
        check_one_of(method, "method", _VALID_METHODS)

        # --- validate alpha -----------------------------------------------
        check_in_range(alpha, "alpha", 0.0, 1.0)

        # --- validate labels ----------------------------------------------
        if labels is not None and len(labels) != len(pv):
            raise ValueError(
                format_error(
                    f"`labels` must have the same length as `pvalues`.",
                    f"pvalues has {len(pv)} elements, labels has {len(labels)} elements.",
                    "provide one label per p-value.",
                )
            )

        self._pvalues = pv
        self._method = method
        self._alpha = alpha
        self._labels = list(labels) if labels is not None else None

    # ─── public ──────────────────────────────────────────────────────

    def run(self) -> CorrectionResult:
        """Apply the correction and return adjusted p-values.

        Returns
        -------
        CorrectionResult
            Frozen dataclass with original p-values, adjusted p-values,
            rejection decisions, and metadata.
        """
        n = len(self._pvalues)

        if self._method == "bonferroni":
            adjusted = self._bonferroni(self._pvalues, n)
        elif self._method == "holm":
            adjusted = self._holm(self._pvalues, n)
        elif self._method == "bh":
            adjusted = self._bh(self._pvalues, n)
        elif self._method == "by":
            adjusted = self._by(self._pvalues, n)
        else:  # pragma: no cover
            raise ValueError(f"Unknown method: {self._method}")

        rejected = [bool(a < self._alpha) for a in adjusted]

        return CorrectionResult(
            pvalues=self._pvalues.tolist(),
            adjusted_pvalues=[float(a) for a in adjusted],
            rejected=rejected,
            alpha=self._alpha,
            method=_METHOD_LABELS[self._method],
            n_rejected=sum(rejected),
            n_tests=n,
            labels=self._labels,
        )

    # ─── correction algorithms ───────────────────────────────────────

    @staticmethod
    def _bonferroni(pvalues: np.ndarray, n: int) -> np.ndarray:
        """Bonferroni correction: adjusted_i = min(p_i * n, 1.0)."""
        return np.minimum(pvalues * n, 1.0)

    @staticmethod
    def _holm(pvalues: np.ndarray, n: int) -> np.ndarray:
        """Holm step-down correction."""
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]

        # adjusted = p * (n - rank) where rank is 0-indexed
        adjusted = sorted_p * (n - np.arange(n))

        # Enforce monotonicity (non-decreasing)
        for i in range(1, n):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])

        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)

        # Map back to original order
        result = np.empty(n)
        result[order] = adjusted
        return result

    @staticmethod
    def _bh(pvalues: np.ndarray, n: int) -> np.ndarray:
        """Benjamini-Hochberg (BH) correction for FDR control."""
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]

        # adjusted = p * n / (rank + 1) where rank is 0-indexed
        ranks = np.arange(n) + 1  # 1-indexed for the formula
        adjusted = sorted_p * n / ranks

        # Enforce monotonicity going backwards (non-decreasing when sorted)
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)

        # Map back to original order
        result = np.empty(n)
        result[order] = adjusted
        return result

    @staticmethod
    def _by(pvalues: np.ndarray, n: int) -> np.ndarray:
        """Benjamini-Yekutieli (BY) correction for FDR under dependence."""
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]

        # Correction factor c_n = sum(1/i for i in 1..n)
        c_n = np.sum(1.0 / np.arange(1, n + 1))

        # adjusted = p * n * c_n / (rank + 1)
        ranks = np.arange(n) + 1
        adjusted = sorted_p * n * c_n / ranks

        # Enforce monotonicity going backwards
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)

        # Map back to original order
        result = np.empty(n)
        result[order] = adjusted
        return result
