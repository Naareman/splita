"""Adaptive enrichment design (Simon & Simon 2013).

Mid-experiment population selection based on subgroup treatment response.
If a treatment works in subgroup A but not subgroup B, stop recruiting
from subgroup B and enrich enrollment from subgroup A.
"""

from __future__ import annotations

import numpy as np

from splita._types import EnrichmentResult
from splita._validation import format_error


class AdaptiveEnrichment:
    """Adaptive enrichment design for subgroup selection.

    Implements a staged design where at each interim analysis, subgroups
    with insufficient treatment response are dropped from further
    enrollment.  This increases power for the remaining subgroups.

    Parameters
    ----------
    futility_threshold : float, default 0.2
        Minimum z-score (absolute) to keep a subgroup.  Subgroups with
        a z-score below this threshold are dropped.
    alpha : float, default 0.05
        Significance level for subgroup tests.

    Examples
    --------
    >>> enricher = AdaptiveEnrichment(futility_threshold=0.3)
    >>> result = enricher.update({
    ...     "young": (2.0, 0.5),
    ...     "old": (0.1, 0.8),
    ... })
    >>> "young" in result.selected_subgroups
    True
    >>> "old" in result.dropped_subgroups
    True
    """

    def __init__(
        self,
        *,
        futility_threshold: float = 0.2,
        alpha: float = 0.05,
    ) -> None:
        if futility_threshold < 0:
            raise ValueError(
                format_error(
                    f"`futility_threshold` must be >= 0, got {futility_threshold}.",
                    "threshold is the minimum z-score to keep a subgroup.",
                    "set to 0 to never drop subgroups.",
                )
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                format_error(
                    f"`alpha` must be in (0, 1), got {alpha}.",
                    "alpha controls the significance level.",
                    "typical values are 0.05, 0.01, or 0.10.",
                )
            )

        self._futility_threshold = futility_threshold
        self._alpha = alpha
        self._stage = 0
        self._selected: list[str] = []
        self._dropped: list[str] = []
        self._history: list[dict[str, float]] = []

    def update(
        self,
        subgroup_results: dict[str, tuple[float, float]],
    ) -> EnrichmentResult:
        """Perform an interim analysis and decide which subgroups to keep.

        Parameters
        ----------
        subgroup_results : dict[str, tuple[float, float]]
            Mapping from subgroup name to ``(effect_estimate, standard_error)``.
            Both values must be finite numbers.  Standard errors must be positive.

        Returns
        -------
        EnrichmentResult
            Which subgroups are selected, dropped, and enrichment ratios.

        Raises
        ------
        ValueError
            If *subgroup_results* is empty or contains invalid values.
        TypeError
            If *subgroup_results* is not a dict.
        """
        if not isinstance(subgroup_results, dict):
            raise TypeError(
                format_error(
                    "`subgroup_results` must be a dict.",
                    f"got type {type(subgroup_results).__name__}.",
                    "pass a dict mapping subgroup name to (effect, se).",
                )
            )

        if len(subgroup_results) == 0:
            raise ValueError(
                format_error(
                    "`subgroup_results` can't be empty.",
                    "received a dict with 0 subgroups.",
                    "provide at least one subgroup with (effect, se).",
                )
            )

        # Validate entries
        for name, val in subgroup_results.items():
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(
                    format_error(
                        f"`subgroup_results['{name}']` must be a (effect, se) tuple.",
                        f"got {val!r}.",
                    )
                )
            effect, se = val
            if not np.isfinite(effect):
                raise ValueError(
                    format_error(
                        f"Effect for subgroup '{name}' must be finite.",
                        f"got {effect}.",
                    )
                )
            if not np.isfinite(se) or se <= 0:
                raise ValueError(
                    format_error(
                        f"Standard error for subgroup '{name}' must be positive and finite.",
                        f"got {se}.",
                        "ensure you have enough observations to estimate SE.",
                    )
                )

        self._stage += 1

        # Filter to only consider subgroups not already dropped
        active_subgroups = {
            name: val for name, val in subgroup_results.items() if name not in self._dropped
        }

        if len(active_subgroups) == 0:  # pragma: no cover
            # All subgroups previously dropped — re-evaluate all
            active_subgroups = subgroup_results

        # Compute z-scores and enrichment ratios
        enrichment_ratios: dict[str, float] = {}
        newly_selected: list[str] = []
        newly_dropped: list[str] = []

        for name, (effect, se) in active_subgroups.items():
            z = abs(effect / se)
            enrichment_ratios[name] = float(z)

            if z >= self._futility_threshold:
                newly_selected.append(name)
            else:
                newly_dropped.append(name)

        # Update cumulative tracking
        for d in newly_dropped:
            if d not in self._dropped:
                self._dropped.append(d)

        self._selected = newly_selected
        self._history.append(enrichment_ratios)

        return EnrichmentResult(
            selected_subgroups=list(newly_selected),
            dropped_subgroups=list(self._dropped),
            enrichment_ratios=enrichment_ratios,
            stage=self._stage,
        )

    def result(self) -> EnrichmentResult:
        """Return the current enrichment state.

        Returns
        -------
        EnrichmentResult
            Current state of subgroup selection.

        Raises
        ------
        ValueError
            If :meth:`update` has never been called.
        """
        if self._stage == 0:
            raise ValueError(
                format_error(
                    "No interim analyses have been performed yet.",
                    "call update() with subgroup results first.",
                )
            )

        return EnrichmentResult(
            selected_subgroups=list(self._selected),
            dropped_subgroups=list(self._dropped),
            enrichment_ratios=self._history[-1] if self._history else {},
            stage=self._stage,
        )
