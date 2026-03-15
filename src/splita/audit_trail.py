"""Immutable experiment audit records.

Provides :func:`audit_trail` which creates a tamper-evident record of an
experiment analysis by hashing the result, parameters, and metadata into
a single verifiable record.

Examples
--------
>>> from splita._types import ExperimentResult
>>> from splita import audit_trail
>>> r = ExperimentResult(
...     control_mean=0.10, treatment_mean=0.12,
...     lift=0.02, relative_lift=0.2, pvalue=0.03,
...     statistic=2.1, ci_lower=0.002, ci_upper=0.038,
...     significant=True, alpha=0.05, method="ztest",
...     metric="conversion", control_n=1000,
...     treatment_n=1000, power=0.65, effect_size=0.06,
... )
>>> record = audit_trail(r, analyst="alice")
>>> len(record.result_hash)
64
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from splita._types import AuditRecord


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def audit_trail(
    result: object,
    *,
    data_hash: str | None = None,
    parameters: dict[str, Any] | None = None,
    analyst: str | None = None,
    timestamp: str | None = None,
) -> AuditRecord:
    """Create an immutable, hashable record of an experiment analysis.

    The record contains SHA-256 hashes that prove the analysis was not
    tampered with after the fact.

    Parameters
    ----------
    result : any splita result object
        Must have a ``to_json()`` method (all splita result dataclasses do).
    data_hash : str or None, default None
        Pre-computed SHA-256 hex digest of the input data.  If not provided,
        the record will have ``data_hash=None``.
    parameters : dict or None, default None
        Analysis parameters to record (e.g., alpha, method, metric).
        If ``None``, defaults to an empty dict.
    analyst : str or None, default None
        Name or identifier of the analyst.
    timestamp : str or None, default None
        ISO-8601 timestamp.  If ``None``, the current UTC time is used.

    Returns
    -------
    AuditRecord
        Immutable record with result hash, data hash, parameters,
        analyst, timestamp, and a record-level hash.

    Raises
    ------
    ValueError
        If the result object does not have a ``to_json()`` method.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.03,
    ...     statistic=2.1, ci_lower=0.002, ci_upper=0.038,
    ...     significant=True, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=1000,
    ...     treatment_n=1000, power=0.65, effect_size=0.06,
    ... )
    >>> record = audit_trail(r, analyst="alice")
    >>> len(record.result_hash)
    64
    >>> len(record.record_hash)
    64
    """
    if not hasattr(result, "to_json"):
        raise ValueError(
            "Result object must have a `to_json()` method.\n"
            "  Detail: audit_trail needs to serialize the result to compute "
            "a hash.\n"
            "  Hint: pass a splita result object (e.g., ExperimentResult)."
        )

    result_json = result.to_json()  # type: ignore[attr-defined]
    result_hash = _sha256(result_json)

    if parameters is None:
        parameters = {}

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    # Compute record hash from all fields
    record_payload = json.dumps(
        {
            "result_hash": result_hash,
            "data_hash": data_hash,
            "parameters": parameters,
            "analyst": analyst,
            "timestamp": timestamp,
        },
        sort_keys=True,
        default=str,
    )
    record_hash = _sha256(record_payload)

    return AuditRecord(
        result_hash=result_hash,
        data_hash=data_hash,
        parameters=parameters,
        analyst=analyst,
        timestamp=timestamp,
        record_hash=record_hash,
    )
