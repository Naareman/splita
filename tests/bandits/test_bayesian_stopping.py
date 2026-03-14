"""Tests for BayesianStopping stub (planned for v0.2.0)."""

from __future__ import annotations

import pytest

from splita.bandits.bayesian_stopping import BayesianStopping


class TestBayesianStoppingStub:
    """BayesianStopping is a planned stub — verify it raises NotImplementedError."""

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match=r"planned for splita v0.2.0"):
            BayesianStopping()

    def test_raises_with_args(self):
        with pytest.raises(NotImplementedError, match=r"BayesianStopping"):
            BayesianStopping(n_arms=3)
