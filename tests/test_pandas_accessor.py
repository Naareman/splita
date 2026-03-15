"""Tests for the pandas DataFrame accessor."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

# Import to register the accessor
import splita.integrations.pandas_accessor  # noqa: F401
from splita._types import ExperimentResult, SRMResult


class TestSplitaAccessor:
    """Tests for df.splita accessor."""

    def test_experiment_continuous(self) -> None:
        """Accessor runs a continuous experiment on DataFrame columns."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "control": rng.normal(10, 2, 100),
            "treatment": rng.normal(10.5, 2, 100),
        })
        result = df.splita.experiment("control", "treatment")
        assert isinstance(result, ExperimentResult)
        assert result.method == "ttest"
        assert result.control_n == 100
        assert result.treatment_n == 100

    def test_experiment_conversion(self) -> None:
        """Accessor detects conversion metric on binary columns."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "control": rng.choice([0, 1], size=200, p=[0.9, 0.1]),
            "treatment": rng.choice([0, 1], size=200, p=[0.85, 0.15]),
        })
        result = df.splita.experiment("control", "treatment")
        assert isinstance(result, ExperimentResult)
        assert result.metric == "conversion"

    def test_experiment_with_nans(self) -> None:
        """NaN values are dropped before analysis."""
        df = pd.DataFrame({
            "control": [1.0, 2.0, np.nan, 3.0, 4.0],
            "treatment": [2.0, 3.0, 4.0, np.nan, 5.0],
        })
        result = df.splita.experiment("control", "treatment")
        assert isinstance(result, ExperimentResult)
        assert result.control_n == 4
        assert result.treatment_n == 4

    def test_experiment_with_kwargs(self) -> None:
        """Extra kwargs are forwarded to Experiment."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "control": rng.normal(10, 2, 50),
            "treatment": rng.normal(10.5, 2, 50),
        })
        result = df.splita.experiment("control", "treatment", method="bootstrap",
                                       n_bootstrap=500, random_state=42)
        assert isinstance(result, ExperimentResult)
        assert result.method == "bootstrap"

    def test_experiment_missing_column(self) -> None:
        """KeyError raised for nonexistent column."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError, match="not_here"):
            df.splita.experiment("a", "not_here")

    def test_experiment_missing_control_column(self) -> None:
        """KeyError raised for nonexistent control column."""
        df = pd.DataFrame({"b": [1, 2, 3]})
        with pytest.raises(KeyError, match="missing"):
            df.splita.experiment("missing", "b")

    def test_check_srm_equal_split(self) -> None:
        """SRM check passes for balanced groups."""
        df = pd.DataFrame({"group": ["A"] * 500 + ["B"] * 500})
        result = df.splita.check_srm("group")
        assert isinstance(result, SRMResult)
        assert result.passed is True

    def test_check_srm_imbalanced(self) -> None:
        """SRM check detects severe imbalance."""
        df = pd.DataFrame({"group": ["A"] * 300 + ["B"] * 700})
        result = df.splita.check_srm("group")
        assert isinstance(result, SRMResult)
        assert result.passed is False

    def test_check_srm_custom_fractions(self) -> None:
        """SRM check with custom expected fractions."""
        df = pd.DataFrame({"group": ["A"] * 300 + ["B"] * 700})
        result = df.splita.check_srm("group", expected_fractions=[0.3, 0.7])
        assert isinstance(result, SRMResult)
        assert result.passed is True

    def test_check_srm_missing_column(self) -> None:
        """KeyError raised for nonexistent group column."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError, match="missing"):
            df.splita.check_srm("missing")
