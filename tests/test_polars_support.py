"""Tests for polars Series support."""

from __future__ import annotations

import numpy as np
import pytest

polars = pytest.importorskip("polars")

from splita import Experiment
from splita._types import ExperimentResult
from splita._validation import check_array_like
from splita.integrations.polars_support import from_polars


class TestPolarsSupport:
    """Tests for polars integration."""

    def test_check_array_like_accepts_polars_series(self) -> None:
        """check_array_like converts a polars Series to numpy."""
        s = polars.Series("data", [1.0, 2.0, 3.0, 4.0])
        arr = check_array_like(s, "test")
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])

    def test_experiment_with_polars_series(self) -> None:
        """Experiment accepts polars Series as input."""
        ctrl = polars.Series("ctrl", [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        trt = polars.Series("trt", [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        result = Experiment(ctrl, trt).run()
        assert isinstance(result, ExperimentResult)
        assert result.metric == "conversion"

    def test_from_polars_helper(self) -> None:
        """from_polars converts a Series to numpy array."""
        s = polars.Series("x", [10, 20, 30])
        arr = from_polars(s)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [10, 20, 30])

    def test_from_polars_rejects_non_series(self) -> None:
        """from_polars raises TypeError for non-Series input."""
        with pytest.raises(TypeError, match="Expected a polars Series"):
            from_polars([1, 2, 3])

    def test_polars_series_with_nans(self) -> None:
        """Polars Series with NaN values are handled correctly."""
        s = polars.Series("data", [1.0, float("nan"), 3.0, 4.0, 5.0])
        arr = check_array_like(s, "test")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 4  # NaN dropped
