"""In-memory experiment registry for tracking active experiments.

Provides a lightweight way to register, deregister, and query experiments
by date range without requiring an external database.
"""

from __future__ import annotations

from datetime import date, datetime

from splita._validation import format_error


def _parse_date(s: str) -> date:
    """Parse an ISO-format date string (YYYY-MM-DD).

    Parameters
    ----------
    s : str
        Date string in ISO 8601 format.

    Returns
    -------
    date
        Parsed date object.

    Raises
    ------
    ValueError
        If the string is not a valid ISO date.
    """
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            format_error(
                f"invalid date string {s!r}.",
                detail="expected ISO 8601 format YYYY-MM-DD.",
                hint="e.g. '2026-03-14'.",
            )
        ) from exc


class ExperimentRegistry:
    """Simple in-memory experiment registry.

    Tracks active experiments and their metadata.  Useful for
    coordinating multiple experiments and detecting conflicts via
    :class:`~splita.governance.ConflictDetector`.

    Examples
    --------
    >>> reg = ExperimentRegistry()
    >>> exp = reg.register("test_v1", start_date="2026-03-01",
    ...                    end_date="2026-03-31", traffic_fraction=0.5)
    >>> exp["name"]
    'test_v1'
    >>> reg.list_active(as_of="2026-03-15")
    [{'name': 'test_v1', ...}]
    """

    def __init__(self) -> None:
        self._experiments: dict[str, dict] = {}

    def register(
        self,
        name: str,
        *,
        start_date: str,
        end_date: str,
        traffic_fraction: float = 1.0,
        metrics: list[str] | None = None,
        segments: list[str] | None = None,
    ) -> dict:
        """Register a new experiment.

        Parameters
        ----------
        name : str
            Unique experiment name.
        start_date : str
            Start date in ISO 8601 format (YYYY-MM-DD).
        end_date : str
            End date in ISO 8601 format (YYYY-MM-DD).
        traffic_fraction : float, default 1.0
            Fraction of traffic allocated to this experiment, in (0, 1].
        metrics : list of str, optional
            Metric names being tracked.
        segments : list of str, optional
            User segment names targeted by this experiment.

        Returns
        -------
        dict
            The registered experiment record.

        Raises
        ------
        ValueError
            If *name* is already registered, dates are invalid, or
            *traffic_fraction* is out of range.
        """
        # ── validate name ──
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                format_error(
                    "`name` must be a non-empty string.",
                    hint="provide a descriptive experiment name.",
                )
            )

        if name in self._experiments:
            raise ValueError(
                format_error(
                    f"experiment {name!r} is already registered.",
                    detail="experiment names must be unique.",
                    hint="use deregister() first or choose a different name.",
                )
            )

        # ── validate dates ──
        start = _parse_date(start_date)
        end = _parse_date(end_date)

        if end < start:
            raise ValueError(
                format_error(
                    f"`end_date` ({end_date}) must be >= `start_date` ({start_date}).",
                    detail="the experiment cannot end before it starts.",
                )
            )

        # ── validate traffic_fraction ──
        if not (0.0 < traffic_fraction <= 1.0):
            raise ValueError(
                format_error(
                    f"`traffic_fraction` must be in (0, 1], got {traffic_fraction}.",
                    hint="typical values are 0.1 to 1.0.",
                )
            )

        experiment = {
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "traffic_fraction": traffic_fraction,
            "metrics": metrics or [],
            "segments": segments or [],
        }

        self._experiments[name] = experiment
        return experiment

    def deregister(self, name: str) -> None:
        """Remove an experiment from the registry.

        Parameters
        ----------
        name : str
            Name of the experiment to remove.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in self._experiments:
            raise KeyError(
                format_error(
                    f"experiment {name!r} is not registered.",
                    hint="check the name or call list_active() to see current experiments.",
                )
            )
        del self._experiments[name]

    def list_active(self, as_of: str | None = None) -> list[dict]:
        """List experiments that are active on a given date.

        Parameters
        ----------
        as_of : str or None, default None
            ISO date string.  If ``None``, returns all registered
            experiments.

        Returns
        -------
        list of dict
            Experiment records active on the given date, sorted by name.
        """
        if as_of is None:
            return sorted(self._experiments.values(), key=lambda e: e["name"])

        target = _parse_date(as_of)
        active = []
        for exp in self._experiments.values():
            start = _parse_date(exp["start_date"])
            end = _parse_date(exp["end_date"])
            if start <= target <= end:
                active.append(exp)
        return sorted(active, key=lambda e: e["name"])

    def get(self, name: str) -> dict:
        """Get a single experiment by name.

        Parameters
        ----------
        name : str
            Experiment name.

        Returns
        -------
        dict
            The experiment record.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in self._experiments:
            raise KeyError(
                format_error(
                    f"experiment {name!r} is not registered.",
                    hint="check the name or call list_active() to see current experiments.",
                )
            )
        return self._experiments[name]
