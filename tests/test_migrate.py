"""Tests for the migrate module."""

from __future__ import annotations

import math

import pytest

from splita._types import ExperimentResult
from splita.integrations.migrate import migrate_from


class TestMigrateGrowthBook:
    """Tests for GrowthBook migration."""

    def test_basic_growthbook(self) -> None:
        """Migrates a standard GrowthBook result."""
        data = {
            "chance_to_win": 0.95,
            "effect": 0.02,
            "ci_lower": 0.005,
            "ci_upper": 0.035,
            "control_mean": 0.10,
            "treatment_mean": 0.12,
            "control_n": 5000,
            "treatment_n": 5000,
        }
        result = migrate_from(data, platform="growthbook")
        assert isinstance(result, ExperimentResult)
        assert result.lift == 0.02
        assert result.ci_lower == 0.005
        assert result.ci_upper == 0.035
        assert result.method == "growthbook_bayesian"
        assert result.control_n == 5000

    def test_growthbook_significant(self) -> None:
        """High chance_to_win maps to significant."""
        data = {
            "chance_to_win": 0.99,
            "effect": 0.05,
            "ci_lower": 0.02,
            "ci_upper": 0.08,
        }
        result = migrate_from(data, platform="growthbook")
        assert result.significant is True

    def test_growthbook_not_significant(self) -> None:
        """Low chance_to_win maps to not significant."""
        data = {
            "chance_to_win": 0.55,
            "effect": 0.001,
            "ci_lower": -0.01,
            "ci_upper": 0.012,
        }
        result = migrate_from(data, platform="growthbook")
        assert result.significant is False

    def test_growthbook_missing_keys(self) -> None:
        """ValueError for missing required keys."""
        with pytest.raises(ValueError, match="Missing required keys"):
            migrate_from({"chance_to_win": 0.95}, platform="growthbook")


class TestMigrateStatsig:
    """Tests for Statsig migration."""

    def test_basic_statsig(self) -> None:
        """Migrates a standard Statsig result."""
        data = {
            "p_value": 0.03,
            "effect_size": 0.015,
            "ci_lower": 0.002,
            "ci_upper": 0.028,
            "control_mean": 0.10,
            "treatment_mean": 0.115,
            "control_n": 10000,
            "treatment_n": 10000,
        }
        result = migrate_from(data, platform="statsig")
        assert isinstance(result, ExperimentResult)
        assert result.pvalue == 0.03
        assert result.significant is True
        assert result.method == "statsig_frequentist"

    def test_statsig_missing_keys(self) -> None:
        """ValueError for missing required keys."""
        with pytest.raises(ValueError, match="Missing required keys"):
            migrate_from({"p_value": 0.05}, platform="statsig")


class TestMigrateGeneric:
    """Tests for generic migration."""

    def test_basic_generic(self) -> None:
        """Migrates a generic dict with effect/pvalue/ci."""
        data = {
            "effect": 2.5,
            "pvalue": 0.01,
            "ci_lower": 0.5,
            "ci_upper": 4.5,
            "control_mean": 25.0,
            "treatment_mean": 27.5,
        }
        result = migrate_from(data, platform="generic")
        assert isinstance(result, ExperimentResult)
        assert result.lift == 2.5
        assert result.pvalue == 0.01
        assert result.significant is True

    def test_generic_minimal_keys(self) -> None:
        """Generic migration works with only the required keys."""
        data = {
            "effect": 1.0,
            "pvalue": 0.5,
            "ci_lower": -1.0,
            "ci_upper": 3.0,
        }
        result = migrate_from(data, platform="generic")
        assert isinstance(result, ExperimentResult)
        assert result.control_n == 0  # default

    def test_generic_missing_keys(self) -> None:
        """ValueError for missing required keys."""
        with pytest.raises(ValueError, match="Missing required keys"):
            migrate_from({"effect": 1.0}, platform="generic")


class TestMigrateUnsupported:
    """Tests for unsupported platform."""

    def test_unsupported_platform(self) -> None:
        """ValueError for unknown platform name."""
        with pytest.raises(ValueError, match="Unsupported platform"):
            migrate_from({}, platform="unknown_platform")
