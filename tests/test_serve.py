"""Tests for the REST API wrapper."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from splita.serve import _create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a test client for the splita API."""
    app = _create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health(self, client: TestClient) -> None:
        """Health endpoint returns OK."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "splita"


class TestExperimentEndpoint:
    """Tests for POST /experiment."""

    def test_experiment_continuous(self, client: TestClient) -> None:
        """Experiment endpoint runs a continuous test."""
        resp = client.post("/experiment", json={
            "control": [1.0, 2.0, 3.0, 4.0, 5.0],
            "treatment": [2.0, 3.0, 4.0, 5.0, 6.0],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "pvalue" in data
        assert data["method"] == "ttest"

    def test_experiment_conversion(self, client: TestClient) -> None:
        """Experiment endpoint detects conversion metric."""
        resp = client.post("/experiment", json={
            "control": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            "treatment": [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["metric"] == "conversion"

    def test_experiment_with_params(self, client: TestClient) -> None:
        """Experiment endpoint accepts custom parameters."""
        resp = client.post("/experiment", json={
            "control": [1.0, 2.0, 3.0, 4.0, 5.0],
            "treatment": [2.0, 3.0, 4.0, 5.0, 6.0],
            "alpha": 0.01,
            "alternative": "greater",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["alpha"] == 0.01

    def test_experiment_invalid_input(self, client: TestClient) -> None:
        """Invalid input returns 422."""
        resp = client.post("/experiment", json={
            "control": [1.0],
            "treatment": [2.0],
        })
        assert resp.status_code == 422


class TestSampleSizeEndpoint:
    """Tests for POST /sample-size."""

    def test_sample_size_proportion(self, client: TestClient) -> None:
        """Sample size for proportion test."""
        resp = client.post("/sample-size", json={
            "test_type": "proportion",
            "baseline": 0.10,
            "mde": 0.02,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "n_per_variant" in data
        assert data["n_per_variant"] > 0

    def test_sample_size_mean(self, client: TestClient) -> None:
        """Sample size for mean test."""
        resp = client.post("/sample-size", json={
            "test_type": "mean",
            "baseline": 25.0,
            "mde": 2.0,
            "baseline_std": 10.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_per_variant"] > 0

    def test_sample_size_mean_missing_std(self, client: TestClient) -> None:
        """Missing baseline_std for mean test returns 422."""
        resp = client.post("/sample-size", json={
            "test_type": "mean",
            "baseline": 25.0,
            "mde": 2.0,
        })
        assert resp.status_code == 422

    def test_sample_size_unsupported_type(self, client: TestClient) -> None:
        """Unsupported test_type returns 422."""
        resp = client.post("/sample-size", json={
            "test_type": "ratio",
            "baseline": 0.5,
            "mde": 0.01,
        })
        assert resp.status_code == 422


class TestSRMCheckEndpoint:
    """Tests for POST /srm-check."""

    def test_srm_balanced(self, client: TestClient) -> None:
        """Balanced groups pass SRM check."""
        resp = client.post("/srm-check", json={
            "observed": [5000, 5000],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True

    def test_srm_imbalanced(self, client: TestClient) -> None:
        """Imbalanced groups fail SRM check."""
        resp = client.post("/srm-check", json={
            "observed": [3000, 7000],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False

    def test_srm_custom_fractions(self, client: TestClient) -> None:
        """Custom fractions are respected."""
        resp = client.post("/srm-check", json={
            "observed": [3000, 7000],
            "expected_fractions": [0.3, 0.7],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True
