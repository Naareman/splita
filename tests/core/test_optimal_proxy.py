"""Tests for OptimalProxyMetrics."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.optimal_proxy import OptimalProxyMetrics


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def data(rng):
    n = 200
    ns = rng.normal(10, 2, n)
    candidates = np.column_stack([
        ns + rng.normal(0, 1, n),
        rng.normal(5, 1, n),
        ns * 0.5 + rng.normal(0, 0.5, n),
    ])
    return candidates, ns


class TestFit:
    def test_learns_weights(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        assert len(proxy.weights_) == 3

    def test_returns_self(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        result = proxy.fit(candidates, ns)
        assert result is proxy


class TestTransform:
    def test_produces_output(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        values = proxy.transform(candidates)
        assert len(values) == len(ns)

    def test_transform_before_fit(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        with pytest.raises(RuntimeError, match="fitted"):
            proxy.transform(candidates)


class TestResult:
    def test_high_correlation(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        result = proxy.result(candidates, ns)
        assert result.correlation_with_north_star > 0.5

    def test_result_weights(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        result = proxy.result(candidates, ns)
        assert len(result.weights) == 3

    def test_result_proxy_values(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        result = proxy.result(candidates, ns)
        assert len(result.optimal_proxy_values) == len(ns)

    def test_to_dict(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        result = proxy.result(candidates, ns)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "weights" in d

    def test_repr(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        result = proxy.result(candidates, ns)
        assert "ProxyResult" in repr(result)


class TestValidation:
    def test_1d_input(self, rng):
        with pytest.raises(ValueError, match="2-D"):
            proxy = OptimalProxyMetrics()
            proxy.fit(rng.normal(0, 1, 50), rng.normal(0, 1, 50))

    def test_mismatched_rows(self, rng):
        with pytest.raises(ValueError, match="same number"):
            proxy = OptimalProxyMetrics()
            proxy.fit(rng.normal(0, 1, (50, 3)), rng.normal(0, 1, 40))

    def test_no_columns(self, rng):
        with pytest.raises(ValueError, match="at least 1"):
            proxy = OptimalProxyMetrics()
            proxy.fit(np.empty((50, 0)), rng.normal(0, 1, 50))

    def test_more_cols_than_rows(self, rng):
        with pytest.raises(ValueError, match="more rows"):
            proxy = OptimalProxyMetrics()
            proxy.fit(rng.normal(0, 1, (3, 5)), rng.normal(0, 1, 3))

    def test_transform_wrong_columns(self, data):
        candidates, ns = data
        proxy = OptimalProxyMetrics()
        proxy.fit(candidates, ns)
        with pytest.raises(ValueError, match="columns"):
            proxy.transform(np.column_stack([candidates, candidates[:, 0]]))
