"""Tests for splita.audit_trail."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from splita._types import ExperimentResult
from splita.audit_trail import audit_trail


def _make_result() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.2,
        pvalue=0.03,
        statistic=2.1,
        ci_lower=0.002,
        ci_upper=0.038,
        significant=True,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=1000,
        treatment_n=1000,
        power=0.65,
        effect_size=0.06,
    )


class TestAuditTrailBasic:
    """Basic audit trail functionality."""

    def test_returns_audit_record(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        assert hasattr(record, "result_hash")
        assert hasattr(record, "record_hash")

    def test_result_hash_is_sha256(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        assert len(record.result_hash) == 64  # SHA-256 hex

    def test_record_hash_is_sha256(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        assert len(record.record_hash) == 64

    def test_same_result_same_hash(self) -> None:
        r = _make_result()
        rec1 = audit_trail(r, timestamp="2024-01-01T00:00:00+00:00")
        rec2 = audit_trail(r, timestamp="2024-01-01T00:00:00+00:00")
        assert rec1.result_hash == rec2.result_hash
        assert rec1.record_hash == rec2.record_hash

    def test_different_results_different_hash(self) -> None:
        r1 = _make_result()
        r2 = ExperimentResult(
            control_mean=0.10, treatment_mean=0.15,
            lift=0.05, relative_lift=0.5, pvalue=0.001,
            statistic=3.5, ci_lower=0.02, ci_upper=0.08,
            significant=True, alpha=0.05, method="ztest",
            metric="conversion", control_n=1000,
            treatment_n=1000, power=0.95, effect_size=0.15,
        )
        rec1 = audit_trail(r1, timestamp="2024-01-01T00:00:00+00:00")
        rec2 = audit_trail(r2, timestamp="2024-01-01T00:00:00+00:00")
        assert rec1.result_hash != rec2.result_hash


class TestAuditTrailMetadata:
    """Metadata handling."""

    def test_analyst_stored(self) -> None:
        r = _make_result()
        record = audit_trail(r, analyst="alice")
        assert record.analyst == "alice"

    def test_analyst_default_none(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        assert record.analyst is None

    def test_data_hash_stored(self) -> None:
        r = _make_result()
        data_hash = hashlib.sha256(b"some data").hexdigest()
        record = audit_trail(r, data_hash=data_hash)
        assert record.data_hash == data_hash

    def test_parameters_stored(self) -> None:
        r = _make_result()
        params = {"alpha": 0.05, "method": "ztest"}
        record = audit_trail(r, parameters=params)
        assert record.parameters == params

    def test_default_parameters_empty_dict(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        assert record.parameters == {}

    def test_custom_timestamp(self) -> None:
        r = _make_result()
        ts = "2024-06-15T12:00:00+00:00"
        record = audit_trail(r, timestamp=ts)
        assert record.timestamp == ts

    def test_auto_timestamp_is_iso(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        # Should be parseable as ISO
        dt = datetime.fromisoformat(record.timestamp)
        assert dt.tzinfo is not None


class TestAuditTrailImmutability:
    """Records are frozen."""

    def test_frozen(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        with pytest.raises(AttributeError):
            record.analyst = "bob"  # type: ignore[misc]


class TestAuditTrailValidation:
    """Input validation."""

    def test_no_to_json_raises(self) -> None:
        with pytest.raises(ValueError, match="to_json"):
            audit_trail("not a result")  # type: ignore[arg-type]


class TestAuditTrailSerialization:
    """Serialization."""

    def test_to_dict(self) -> None:
        r = _make_result()
        record = audit_trail(r, analyst="alice")
        d = record.to_dict()
        assert isinstance(d, dict)
        assert "result_hash" in d
        assert d["analyst"] == "alice"

    def test_to_json(self) -> None:
        r = _make_result()
        record = audit_trail(r)
        j = record.to_json()
        assert isinstance(j, str)
        assert "result_hash" in j


class TestAuditTrailRepr:
    """String representation."""

    def test_repr_contains_hash(self) -> None:
        r = _make_result()
        record = audit_trail(r, analyst="alice")
        s = repr(record)
        assert "AuditRecord" in s
        assert "alice" in s
