"""Tests for TriggeredExperiment and InteractionTest (M14)."""

from __future__ import annotations

import numpy as np
import pytest

from splita import (
    InteractionResult,
    InteractionTest,
    TriggeredExperiment,
    TriggeredResult,
)


# ─── TriggeredExperiment tests ───────────────────────────────────


class TestTriggeredExperiment:
    def test_itt_uses_all_data(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(12, 2, 500)
        triggered_trt = np.zeros(500, dtype=bool)
        triggered_trt[:250] = True  # only 50% triggered

        result = TriggeredExperiment(
            ctrl,
            trt,
            treatment_triggered=triggered_trt,
        ).run()

        assert isinstance(result, TriggeredResult)
        assert result.itt_result.control_n == 500
        assert result.itt_result.treatment_n == 500

    def test_per_protocol_uses_triggered_only(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 500)
        trt = rng.normal(12, 2, 500)
        triggered_trt = np.zeros(500, dtype=bool)
        triggered_trt[:250] = True

        result = TriggeredExperiment(
            ctrl,
            trt,
            treatment_triggered=triggered_trt,
        ).run()

        assert result.per_protocol_result.treatment_n == 250

    def test_trigger_rates(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 100)
        trt = rng.normal(12, 2, 100)
        triggered_ctrl = np.ones(100, dtype=bool)
        triggered_trt = np.zeros(100, dtype=bool)
        triggered_trt[:70] = True

        result = TriggeredExperiment(
            ctrl,
            trt,
            control_triggered=triggered_ctrl,
            treatment_triggered=triggered_trt,
        ).run()

        assert result.trigger_rate_control == 1.0
        assert result.trigger_rate_treatment == pytest.approx(0.7)

    def test_all_triggered_default(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 100)
        trt = rng.normal(12, 2, 100)

        result = TriggeredExperiment(ctrl, trt).run()
        assert result.trigger_rate_control == 1.0
        assert result.trigger_rate_treatment == 1.0
        # ITT and PP should be identical
        assert result.itt_result.lift == pytest.approx(
            result.per_protocol_result.lift
        )

    def test_partial_triggering_dilutes_itt(self):
        rng = np.random.default_rng(42)
        n = 1000
        ctrl = rng.normal(10, 1, n)
        # Treatment: triggered users get effect, non-triggered don't
        trt = rng.normal(10, 1, n)
        triggered = rng.random(n) > 0.5
        trt[triggered] += 3.0  # large effect only for triggered

        result = TriggeredExperiment(
            ctrl,
            trt,
            treatment_triggered=triggered,
        ).run()

        # Per-protocol lift should be larger than ITT lift
        assert abs(result.per_protocol_result.lift) > abs(result.itt_result.lift)

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 50)
        trt = rng.normal(12, 2, 50)
        result = TriggeredExperiment(ctrl, trt).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "itt_result" in d
        assert "per_protocol_result" in d
        assert isinstance(d["itt_result"], dict)

    def test_repr(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 50)
        trt = rng.normal(12, 2, 50)
        result = TriggeredExperiment(ctrl, trt).run()
        r = repr(result)
        assert "TriggeredResult" in r


# ─── TriggeredExperiment validation ──────────────────────────────


class TestTriggeredValidation:
    def test_mismatched_trigger_mask(self):
        with pytest.raises(ValueError, match="same length"):
            TriggeredExperiment(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
                control_triggered=np.array([True, False]),  # wrong length
            ).run()

    def test_too_few_triggered(self):
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([4.0, 5.0, 6.0])
        triggered = np.array([True, False, False])
        with pytest.raises(ValueError, match="at least 2"):
            TriggeredExperiment(
                ctrl,
                trt,
                treatment_triggered=triggered,
            ).run()

    def test_too_few_triggered_control(self):
        """Line 125: fewer than 2 triggered control observations."""
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        ctrl_triggered = np.array([True, False, False, False, False])
        trt_triggered = np.array([True, True, True, True, True])
        with pytest.raises(ValueError, match="at least 2"):
            TriggeredExperiment(
                ctrl, trt,
                control_triggered=ctrl_triggered,
                treatment_triggered=trt_triggered,
            ).run()

    def test_control_too_short(self):
        with pytest.raises(ValueError, match="at least"):
            TriggeredExperiment(
                np.array([1.0]),
                np.array([1.0, 2.0]),
            ).run()


# ─── InteractionTest tests ───────────────────────────────────────


class TestInteractionTest:
    def test_detects_interaction(self):
        rng = np.random.default_rng(42)
        n = 500
        # Segment A: large effect, Segment B: no effect
        ctrl_a = rng.normal(10, 1, n)
        ctrl_b = rng.normal(10, 1, n)
        trt_a = rng.normal(15, 1, n)   # +5 effect
        trt_b = rng.normal(10, 1, n)   # 0 effect

        ctrl = np.concatenate([ctrl_a, ctrl_b])
        trt = np.concatenate([trt_a, trt_b])
        segs = np.array(
            ["A"] * n + ["B"] * n + ["A"] * n + ["B"] * n
        )

        result = InteractionTest(ctrl, trt, segments=segs).run()
        assert isinstance(result, InteractionResult)
        assert result.has_interaction
        assert result.interaction_pvalue < 0.05
        assert result.strongest_segment == "A"

    def test_no_interaction_uniform_effect(self):
        rng = np.random.default_rng(42)
        n = 300
        ctrl = rng.normal(10, 1, 2 * n)
        trt = rng.normal(11, 1, 2 * n)  # uniform +1 effect
        segs = np.array(["A"] * n + ["B"] * n + ["A"] * n + ["B"] * n)

        result = InteractionTest(ctrl, trt, segments=segs).run()
        assert not result.has_interaction
        assert result.interaction_pvalue > 0.05

    def test_segment_results_structure(self):
        rng = np.random.default_rng(42)
        n = 100
        ctrl = rng.normal(10, 1, 2 * n)
        trt = rng.normal(11, 1, 2 * n)
        segs = np.array(["X"] * n + ["Y"] * n + ["X"] * n + ["Y"] * n)

        result = InteractionTest(ctrl, trt, segments=segs).run()
        assert len(result.segment_results) == 2
        for seg_r in result.segment_results:
            assert "segment" in seg_r
            assert "lift" in seg_r
            assert "pvalue" in seg_r
            assert "ci_lower" in seg_r
            assert "ci_upper" in seg_r

    def test_three_segments(self):
        rng = np.random.default_rng(42)
        n = 100
        ctrl = rng.normal(10, 1, 3 * n)
        trt = rng.normal(11, 1, 3 * n)
        segs = np.array(
            ["A"] * n + ["B"] * n + ["C"] * n
            + ["A"] * n + ["B"] * n + ["C"] * n
        )

        result = InteractionTest(ctrl, trt, segments=segs).run()
        assert len(result.segment_results) == 3

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        n = 50
        ctrl = rng.normal(10, 1, 2 * n)
        trt = rng.normal(11, 1, 2 * n)
        segs = np.array(["A"] * n + ["B"] * n + ["A"] * n + ["B"] * n)
        result = InteractionTest(ctrl, trt, segments=segs).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "has_interaction" in d

    def test_repr(self):
        rng = np.random.default_rng(42)
        n = 50
        ctrl = rng.normal(10, 1, 2 * n)
        trt = rng.normal(11, 1, 2 * n)
        segs = np.array(["A"] * n + ["B"] * n + ["A"] * n + ["B"] * n)
        result = InteractionTest(ctrl, trt, segments=segs).run()
        r = repr(result)
        assert "InteractionResult" in r


# ─── InteractionTest validation ──────────────────────────────────


class TestInteractionValidation:
    def test_wrong_segments_length(self):
        with pytest.raises(ValueError, match="len\\(control\\) \\+ len\\(treatment\\)"):
            InteractionTest(
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                segments=np.array(["A", "B", "A"]),  # should be 4
            )

    def test_single_segment(self):
        with pytest.raises(ValueError, match="at least 2 unique segments"):
            InteractionTest(
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0, 6.0]),
                segments=np.array(["A", "A", "A", "A", "A", "A"]),
            ).run()

    def test_segment_too_few_control(self):
        """Line 257: segment with <2 control observations."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([4.0, 5.0, 6.0])
        # Segment A: 1 control, 2 treatment; Segment B: 2 control, 1 treatment
        segs = np.array(["A", "B", "B", "A", "A", "B"])
        with pytest.raises(ValueError, match="fewer than 2 control"):
            InteractionTest(ctrl, trt, segments=segs).run()

    def test_segment_too_few_treatment(self):
        """Line 265: segment with <2 treatment observations."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt = np.array([4.0, 5.0, 6.0])
        # Segment A: 2 control, 1 treatment; Segment B: 1 control, 2 treatment
        segs = np.array(["A", "A", "B", "A", "B", "B"])
        with pytest.raises(ValueError, match="fewer than 2 treatment"):
            InteractionTest(ctrl, trt, segments=segs).run()

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="must be in"):
            InteractionTest(
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                segments=np.array(["A", "B", "A", "B"]),
                alpha=2.0,
            )
