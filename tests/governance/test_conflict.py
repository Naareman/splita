"""Tests for ConflictDetector."""

import pytest

from splita.governance.conflict import ConflictDetector, ConflictResult
from splita.governance.registry import ExperimentRegistry


@pytest.fixture
def registry():
    return ExperimentRegistry()


# ─── No conflict cases ──────────────────────────────────────────────


class TestNoConflict:
    def test_no_overlap(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-15")
        registry.register("b", start_date="2026-03-16", end_date="2026-03-31")
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert len(results) == 0

    def test_overlap_but_traffic_fits(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.4)
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.4)
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert len(results) == 1
        assert not results[0].has_conflict

    def test_overlap_different_metrics_and_segments(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.3, metrics=["ctr"], segments=["us"])
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.3, metrics=["revenue"], segments=["eu"])
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert len(results) == 1
        assert not results[0].has_conflict

    def test_single_experiment_no_conflicts(self, registry):
        registry.register("only", start_date="2026-03-01", end_date="2026-03-31")
        detector = ConflictDetector(registry)
        results = detector.check("only")
        assert results == []


# ─── Traffic conflicts ──────────────────────────────────────────────


class TestTrafficConflict:
    def test_traffic_overflow(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.6)
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.6)
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert len(results) == 1
        assert results[0].has_conflict
        assert "traffic" in results[0].conflict_types

    def test_traffic_exactly_one_no_conflict(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.5)
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.5)
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert len(results) == 1
        assert "traffic" not in results[0].conflict_types


# ─── Metric conflicts ───────────────────────────────────────────────


class TestMetricConflict:
    def test_shared_metric(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.3, metrics=["ctr", "revenue"])
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.3, metrics=["ctr", "conversion"])
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert len(results) == 1
        assert results[0].has_conflict
        assert "metric" in results[0].conflict_types

    def test_no_shared_metric(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.3, metrics=["ctr"])
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.3, metrics=["revenue"])
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert "metric" not in results[0].conflict_types


# ─── Segment conflicts ──────────────────────────────────────────────


class TestSegmentConflict:
    def test_shared_segment(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.3, segments=["us", "eu"])
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.3, segments=["eu", "asia"])
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert len(results) == 1
        assert results[0].has_conflict
        assert "segment" in results[0].conflict_types

    def test_no_shared_segment(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.3, segments=["us"])
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.3, segments=["eu"])
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert "segment" not in results[0].conflict_types


# ─── Multiple conflict types ────────────────────────────────────────


class TestMultipleConflicts:
    def test_all_conflict_types(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.7, metrics=["ctr"],
                          segments=["us"])
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.7, metrics=["ctr"],
                          segments=["us"])
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert results[0].has_conflict
        assert "traffic" in results[0].conflict_types
        assert "metric" in results[0].conflict_types
        assert "segment" in results[0].conflict_types


# ─── check_all ───────────────────────────────────────────────────────


class TestCheckAll:
    def test_check_all_finds_conflicts(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.7)
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.7)
        registry.register("c", start_date="2026-05-01", end_date="2026-05-31")
        detector = ConflictDetector(registry)
        results = detector.check_all()
        # a-b overlap, a-c no overlap, b-c no overlap
        assert len(results) == 1
        assert results[0].has_conflict

    def test_check_all_no_duplicates(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.7)
        registry.register("b", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.7)
        detector = ConflictDetector(registry)
        results = detector.check_all()
        # Should have exactly one pair, not two
        assert len(results) == 1

    def test_check_all_empty_registry(self, registry):
        detector = ConflictDetector(registry)
        assert detector.check_all() == []


# ─── Overlap days ────────────────────────────────────────────────────


class TestOverlapDays:
    def test_overlap_days_calculation(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31")
        registry.register("b", start_date="2026-03-25", end_date="2026-04-15")
        detector = ConflictDetector(registry)
        results = detector.check("a")
        # 2026-03-25 to 2026-03-31 = 7 days
        assert results[0].overlap_days == 7

    def test_overlap_single_day(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-15")
        registry.register("b", start_date="2026-03-15", end_date="2026-03-31")
        detector = ConflictDetector(registry)
        results = detector.check("a")
        assert results[0].overlap_days == 1


# ─── ConflictResult ─────────────────────────────────────────────────


class TestConflictResult:
    def test_to_dict(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.7)
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.7)
        detector = ConflictDetector(registry)
        results = detector.check("a")
        d = results[0].to_dict()
        assert "experiment_a" in d
        assert "has_conflict" in d

    def test_repr(self, registry):
        registry.register("a", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.7)
        registry.register("b", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.7)
        detector = ConflictDetector(registry)
        results = detector.check("a")
        r = repr(results[0])
        assert "ConflictResult" in r


# ─── Validation ──────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_registry_type(self):
        with pytest.raises(TypeError, match="ExperimentRegistry"):
            ConflictDetector("not_a_registry")

    def test_check_unknown_experiment(self, registry):
        detector = ConflictDetector(registry)
        with pytest.raises(KeyError, match="not registered"):
            detector.check("nonexistent")

    def test_message_contains_experiment_names(self, registry):
        registry.register("alpha", start_date="2026-03-01", end_date="2026-03-31",
                          traffic_fraction=0.7, metrics=["ctr"])
        registry.register("beta", start_date="2026-03-15", end_date="2026-04-15",
                          traffic_fraction=0.7, metrics=["ctr"])
        detector = ConflictDetector(registry)
        results = detector.check("alpha")
        assert "alpha" in results[0].message
        assert "beta" in results[0].message
