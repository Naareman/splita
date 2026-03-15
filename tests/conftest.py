"""Shared test fixtures for splita."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def conversion_data(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Realistic A/B conversion data (~10% rate, 1000 users per group)."""
    return {
        "control": rng.binomial(1, 0.10, size=1000).astype(float),
        "treatment": rng.binomial(1, 0.12, size=1000).astype(float),
    }


@pytest.fixture()
def revenue_data(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Realistic A/B revenue data (lognormal, 1000 users per group)."""
    return {
        "control": rng.lognormal(mean=2.0, sigma=1.0, size=1000),
        "treatment": rng.lognormal(mean=2.05, sigma=1.0, size=1000),
    }
