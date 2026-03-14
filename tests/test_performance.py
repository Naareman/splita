"""Performance benchmarks for splita.

Each test verifies that a core operation meets the PRD performance targets.
Thresholds are set at 2-3x the target to avoid flaky failures on CI.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

from splita import (
    CUPED,
    Experiment,
    MultipleCorrection,
    SampleSize,
    ThompsonSampler,
    mSPRT,
)


@pytest.mark.slow
def test_experiment_run_1m_under_200ms():
    """Experiment.run() on 1M obs should complete in < 200ms (target 100ms)."""
    rng = np.random.default_rng(42)
    ctrl = rng.normal(10.0, 2.0, size=500_000)
    trt = rng.normal(10.1, 2.0, size=500_000)

    start = time.perf_counter()
    result = Experiment(ctrl, trt).run()
    elapsed = time.perf_counter() - start

    assert result is not None
    assert elapsed < 0.200, f"Took {elapsed:.3f}s, expected < 0.200s"


@pytest.mark.slow
def test_msprt_update_batch_1000_under_10ms():
    """mSPRT.update() per batch of 1000 should complete in < 10ms (target 5ms)."""
    rng = np.random.default_rng(42)
    test = mSPRT(metric="conversion", alpha=0.05)

    ctrl = rng.binomial(1, 0.10, size=1000)
    trt = rng.binomial(1, 0.12, size=1000)

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        state = test.update(ctrl, trt)
    elapsed = time.perf_counter() - start

    assert state is not None
    assert elapsed < 0.010, f"Took {elapsed:.3f}s, expected < 0.010s"


@pytest.mark.slow
def test_thompson_recommend_under_2ms():
    """ThompsonSampler.recommend() should complete in < 2ms (target 1ms)."""
    ts = ThompsonSampler(3, random_state=42)
    # Warm up the sampler with some data
    for arm in range(3):
        for _ in range(50):
            ts.update(arm, float(np.random.default_rng(42).random() < 0.5))

    start = time.perf_counter()
    arm = ts.recommend()
    elapsed = time.perf_counter() - start

    assert 0 <= arm < 3
    assert elapsed < 0.002, f"Took {elapsed:.3f}s, expected < 0.002s"


@pytest.mark.slow
def test_sample_size_for_proportion_under_2ms():
    """SampleSize.for_proportion() should complete in < 2ms (target 1ms)."""
    start = time.perf_counter()
    result = SampleSize.for_proportion(0.10, 0.02)
    elapsed = time.perf_counter() - start

    assert result is not None
    assert elapsed < 0.002, f"Took {elapsed:.3f}s, expected < 0.002s"


@pytest.mark.slow
def test_multiple_correction_100_pvalues_under_2ms():
    """MultipleCorrection.run() with 100 p-values in < 2ms (target 1ms)."""
    rng = np.random.default_rng(42)
    pvalues = rng.uniform(0.0, 1.0, size=100).tolist()

    start = time.perf_counter()
    result = MultipleCorrection(pvalues).run()
    elapsed = time.perf_counter() - start

    assert result is not None
    assert elapsed < 0.002, f"Took {elapsed:.3f}s, expected < 0.002s"


@pytest.mark.slow
def test_cuped_fit_transform_100k_under_100ms():
    """CUPED.fit_transform() on 100K obs in < 100ms (target 50ms)."""
    rng = np.random.default_rng(42)
    n = 50_000
    pre_ctrl = rng.normal(10, 2, size=n)
    pre_trt = rng.normal(10, 2, size=n)
    ctrl = pre_ctrl + rng.normal(0, 1, size=n)
    trt = pre_trt + 0.5 + rng.normal(0, 1, size=n)

    cuped = CUPED()

    start = time.perf_counter()
    ctrl_adj, trt_adj = cuped.fit_transform(ctrl, trt, pre_ctrl, pre_trt)
    elapsed = time.perf_counter() - start

    assert ctrl_adj is not None
    assert trt_adj is not None
    assert elapsed < 0.100, f"Took {elapsed:.3f}s, expected < 0.100s"
