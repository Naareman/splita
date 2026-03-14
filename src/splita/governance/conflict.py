"""Conflict detection for overlapping experiments.

Detects experiments that share traffic, metrics, or segments during
overlapping date ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from splita._types import _DictMixin, _fmt, _line
from splita.governance.registry import ExperimentRegistry, _parse_date


@dataclass(frozen=True)
class ConflictResult(_DictMixin):
    """Result of a conflict check between two experiments.

    Attributes
    ----------
    experiment_a : str
        Name of the first experiment.
    experiment_b : str
        Name of the second experiment.
    has_conflict : bool
        Whether a conflict was detected.
    conflict_types : list[str]
        Types of conflicts found: ``'traffic'``, ``'metric'``,
        ``'segment'``.
    overlap_days : int
        Number of days the date ranges overlap.
    message : str
        Human-readable summary of the conflict.
    """

    experiment_a: str
    experiment_b: str
    has_conflict: bool
    conflict_types: list[str]
    overlap_days: int
    message: str

    def __repr__(self) -> str:
        w = 40
        lines = [
            "ConflictResult",
            _line(w),
            f"  {'experiment_a':<20}{self.experiment_a}",
            f"  {'experiment_b':<20}{self.experiment_b}",
            f"  {'has_conflict':<20}{_fmt(self.has_conflict)}",
            f"  {'conflict_types':<20}{self.conflict_types}",
            f"  {'overlap_days':<20}{self.overlap_days}",
            _line(w),
            f"  {self.message}",
        ]
        return "\n".join(lines)


class ConflictDetector:
    """Detect conflicts between overlapping experiments.

    Two experiments conflict if their date ranges overlap AND any of:

    - Combined traffic fractions exceed 1.0
    - They share one or more metrics
    - They share one or more segments

    Parameters
    ----------
    registry : ExperimentRegistry
        The registry to check for conflicts.

    Examples
    --------
    >>> from splita.governance import ExperimentRegistry, ConflictDetector
    >>> reg = ExperimentRegistry()
    >>> reg.register("exp_a", start_date="2026-03-01",
    ...              end_date="2026-03-31", traffic_fraction=0.6,
    ...              metrics=["ctr"])
    {...}
    >>> reg.register("exp_b", start_date="2026-03-15",
    ...              end_date="2026-04-15", traffic_fraction=0.6,
    ...              metrics=["ctr"])
    {...}
    >>> detector = ConflictDetector(reg)
    >>> result = detector.check("exp_a")
    >>> result[0].has_conflict
    True
    """

    def __init__(self, registry: ExperimentRegistry) -> None:
        if not isinstance(registry, ExperimentRegistry):
            raise TypeError(
                f"`registry` must be an ExperimentRegistry, "
                f"got type {type(registry).__name__}."
            )
        self._registry = registry

    def check(self, name: str) -> list[ConflictResult]:
        """Check a specific experiment against all others.

        Parameters
        ----------
        name : str
            Experiment name to check.

        Returns
        -------
        list of ConflictResult
            One result per other experiment that has a date overlap.
            Only experiments with actual date overlap are included.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        target = self._registry.get(name)
        all_exps = self._registry.list_active()
        results = []

        for other in all_exps:
            if other["name"] == name:
                continue
            result = self._check_pair(target, other)
            if result.overlap_days > 0:
                results.append(result)

        return results

    def check_all(self) -> list[ConflictResult]:
        """Check all pairs of experiments for conflicts.

        Returns
        -------
        list of ConflictResult
            One result per pair with date overlap.  Only conflicting or
            overlapping pairs are included.
        """
        all_exps = self._registry.list_active()
        results = []
        seen: set[tuple[str, str]] = set()

        for i, exp_a in enumerate(all_exps):
            for j, exp_b in enumerate(all_exps):
                if i >= j:
                    continue
                pair_key = (exp_a["name"], exp_b["name"])
                if pair_key in seen:  # pragma: no cover
                    continue
                seen.add(pair_key)

                result = self._check_pair(exp_a, exp_b)
                if result.overlap_days > 0:
                    results.append(result)

        return results

    # ─── private helpers ────────────────────────────────────────────

    @staticmethod
    def _check_pair(exp_a: dict, exp_b: dict) -> ConflictResult:
        """Check a single pair of experiments for conflicts."""
        start_a = _parse_date(exp_a["start_date"])
        end_a = _parse_date(exp_a["end_date"])
        start_b = _parse_date(exp_b["start_date"])
        end_b = _parse_date(exp_b["end_date"])

        # Date overlap
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        overlap_days = max(0, (overlap_end - overlap_start).days + 1)

        if overlap_days == 0:
            return ConflictResult(
                experiment_a=exp_a["name"],
                experiment_b=exp_b["name"],
                has_conflict=False,
                conflict_types=[],
                overlap_days=0,
                message="No date overlap.",
            )

        # Check conflict types
        conflict_types: list[str] = []

        # Traffic conflict
        total_traffic = exp_a["traffic_fraction"] + exp_b["traffic_fraction"]
        if total_traffic > 1.0:
            conflict_types.append("traffic")

        # Metric conflict
        metrics_a = set(exp_a.get("metrics", []))
        metrics_b = set(exp_b.get("metrics", []))
        shared_metrics = metrics_a & metrics_b
        if shared_metrics:
            conflict_types.append("metric")

        # Segment conflict
        segments_a = set(exp_a.get("segments", []))
        segments_b = set(exp_b.get("segments", []))
        shared_segments = segments_a & segments_b
        if shared_segments:
            conflict_types.append("segment")

        has_conflict = len(conflict_types) > 0

        # Build message
        if not has_conflict:
            message = (
                f"{exp_a['name']} and {exp_b['name']} overlap for "
                f"{overlap_days} days but have no conflicts."
            )
        else:
            parts = []
            if "traffic" in conflict_types:
                parts.append(f"traffic overflow ({total_traffic:.0%} > 100%)")
            if "metric" in conflict_types:
                parts.append(f"shared metrics: {sorted(shared_metrics)}")
            if "segment" in conflict_types:
                parts.append(f"shared segments: {sorted(shared_segments)}")
            message = (
                f"{exp_a['name']} and {exp_b['name']} conflict over "
                f"{overlap_days} days: {'; '.join(parts)}."
            )

        return ConflictResult(
            experiment_a=exp_a["name"],
            experiment_b=exp_b["name"],
            has_conflict=has_conflict,
            conflict_types=conflict_types,
            overlap_days=overlap_days,
            message=message,
        )
