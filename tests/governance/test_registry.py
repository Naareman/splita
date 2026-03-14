"""Tests for ExperimentRegistry."""

import pytest

from splita.governance.registry import ExperimentRegistry


# ─── Registration ────────────────────────────────────────────────────


class TestRegister:
    def test_register_basic(self):
        reg = ExperimentRegistry()
        exp = reg.register("test_v1", start_date="2026-03-01", end_date="2026-03-31")
        assert exp["name"] == "test_v1"
        assert exp["start_date"] == "2026-03-01"
        assert exp["end_date"] == "2026-03-31"
        assert exp["traffic_fraction"] == 1.0

    def test_register_with_all_params(self):
        reg = ExperimentRegistry()
        exp = reg.register(
            "test_v2",
            start_date="2026-03-01",
            end_date="2026-03-31",
            traffic_fraction=0.5,
            metrics=["ctr", "revenue"],
            segments=["us", "eu"],
        )
        assert exp["traffic_fraction"] == 0.5
        assert exp["metrics"] == ["ctr", "revenue"]
        assert exp["segments"] == ["us", "eu"]

    def test_register_duplicate_name_raises(self):
        reg = ExperimentRegistry()
        reg.register("dup", start_date="2026-03-01", end_date="2026-03-31")
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup", start_date="2026-04-01", end_date="2026-04-30")

    def test_register_empty_name_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError, match="non-empty string"):
            reg.register("", start_date="2026-03-01", end_date="2026-03-31")

    def test_register_whitespace_name_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError, match="non-empty string"):
            reg.register("   ", start_date="2026-03-01", end_date="2026-03-31")

    def test_register_end_before_start_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError, match="must be >="):
            reg.register("bad", start_date="2026-03-31", end_date="2026-03-01")

    def test_register_invalid_date_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError, match="invalid date"):
            reg.register("bad", start_date="not-a-date", end_date="2026-03-31")

    def test_register_traffic_fraction_zero_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError, match="traffic_fraction"):
            reg.register("bad", start_date="2026-03-01", end_date="2026-03-31",
                         traffic_fraction=0.0)

    def test_register_traffic_fraction_over_one_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError, match="traffic_fraction"):
            reg.register("bad", start_date="2026-03-01", end_date="2026-03-31",
                         traffic_fraction=1.5)

    def test_register_negative_traffic_fraction_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError, match="traffic_fraction"):
            reg.register("bad", start_date="2026-03-01", end_date="2026-03-31",
                         traffic_fraction=-0.1)


# ─── Deregistration ─────────────────────────────────────────────────


class TestDeregister:
    def test_deregister_removes_experiment(self):
        reg = ExperimentRegistry()
        reg.register("rm_me", start_date="2026-03-01", end_date="2026-03-31")
        reg.deregister("rm_me")
        assert reg.list_active() == []

    def test_deregister_unknown_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.deregister("nonexistent")

    def test_deregister_allows_reregister(self):
        reg = ExperimentRegistry()
        reg.register("reuse", start_date="2026-03-01", end_date="2026-03-31")
        reg.deregister("reuse")
        exp = reg.register("reuse", start_date="2026-04-01", end_date="2026-04-30")
        assert exp["start_date"] == "2026-04-01"


# ─── List active ─────────────────────────────────────────────────────


class TestListActive:
    def test_list_active_no_filter(self):
        reg = ExperimentRegistry()
        reg.register("a", start_date="2026-03-01", end_date="2026-03-31")
        reg.register("b", start_date="2026-04-01", end_date="2026-04-30")
        assert len(reg.list_active()) == 2

    def test_list_active_filters_by_date(self):
        reg = ExperimentRegistry()
        reg.register("mar", start_date="2026-03-01", end_date="2026-03-31")
        reg.register("apr", start_date="2026-04-01", end_date="2026-04-30")
        active = reg.list_active(as_of="2026-03-15")
        assert len(active) == 1
        assert active[0]["name"] == "mar"

    def test_list_active_boundary_start_date(self):
        reg = ExperimentRegistry()
        reg.register("exp", start_date="2026-03-01", end_date="2026-03-31")
        assert len(reg.list_active(as_of="2026-03-01")) == 1

    def test_list_active_boundary_end_date(self):
        reg = ExperimentRegistry()
        reg.register("exp", start_date="2026-03-01", end_date="2026-03-31")
        assert len(reg.list_active(as_of="2026-03-31")) == 1

    def test_list_active_outside_range(self):
        reg = ExperimentRegistry()
        reg.register("exp", start_date="2026-03-01", end_date="2026-03-31")
        assert len(reg.list_active(as_of="2026-04-01")) == 0

    def test_list_active_sorted_by_name(self):
        reg = ExperimentRegistry()
        reg.register("z_exp", start_date="2026-03-01", end_date="2026-03-31")
        reg.register("a_exp", start_date="2026-03-01", end_date="2026-03-31")
        active = reg.list_active()
        assert active[0]["name"] == "a_exp"
        assert active[1]["name"] == "z_exp"

    def test_list_active_empty_registry(self):
        reg = ExperimentRegistry()
        assert reg.list_active() == []


# ─── Get ─────────────────────────────────────────────────────────────


class TestGet:
    def test_get_returns_experiment(self):
        reg = ExperimentRegistry()
        reg.register("my_exp", start_date="2026-03-01", end_date="2026-03-31")
        exp = reg.get("my_exp")
        assert exp["name"] == "my_exp"

    def test_get_unknown_raises(self):
        reg = ExperimentRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_same_day_experiment(self):
        reg = ExperimentRegistry()
        exp = reg.register("oneday", start_date="2026-03-15", end_date="2026-03-15")
        assert exp["start_date"] == exp["end_date"]
        assert len(reg.list_active(as_of="2026-03-15")) == 1
