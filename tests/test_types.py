from __future__ import annotations

import math

import pytest

from splita._types import (
    BanditResult,
    BoundaryResult,
    CorrectionResult,
    ExperimentResult,
    GSResult,
    SRMResult,
    SampleSizeResult,
    mSPRTResult,
    mSPRTState,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def experiment_result() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.20,
        pvalue=0.0017,
        statistic=3.14,
        ci_lower=0.0075,
        ci_upper=0.0325,
        significant=True,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=5000,
        treatment_n=5200,
        power=0.92,
        effect_size=0.064,
    )


@pytest.fixture
def sample_size_result() -> SampleSizeResult:
    return SampleSizeResult(
        n_per_variant=5000,
        n_total=10000,
        alpha=0.05,
        power=0.80,
        mde=0.02,
        relative_mde=0.20,
        baseline=0.10,
        metric="conversion",
        effect_size=0.064,
        days_needed=None,
    )


@pytest.fixture
def srm_result() -> SRMResult:
    return SRMResult(
        observed=[4850, 5150],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=9.0,
        pvalue=0.0027,
        passed=False,
        alpha=0.01,
        deviations_pct=[-3.0, 3.0],
        worst_variant=1,
        message="Sample ratio mismatch detected!",
    )


@pytest.fixture
def correction_result() -> CorrectionResult:
    return CorrectionResult(
        pvalues=[0.01, 0.04, 0.20],
        adjusted_pvalues=[0.03, 0.06, 0.20],
        rejected=[True, False, False],
        alpha=0.05,
        method="bonferroni",
        n_rejected=1,
        n_tests=3,
        labels=["metric_a", "metric_b", "metric_c"],
    )


@pytest.fixture
def msprt_state() -> mSPRTState:
    return mSPRTState(
        n_control=1000,
        n_treatment=1000,
        mixture_lr=5.2,
        always_valid_pvalue=0.03,
        always_valid_ci_lower=-0.01,
        always_valid_ci_upper=0.05,
        should_stop=True,
        current_effect_estimate=0.02,
    )


@pytest.fixture
def msprt_result() -> mSPRTResult:
    return mSPRTResult(
        n_control=1000,
        n_treatment=1000,
        mixture_lr=5.2,
        always_valid_pvalue=0.03,
        always_valid_ci_lower=-0.01,
        always_valid_ci_upper=0.05,
        should_stop=True,
        current_effect_estimate=0.02,
        stopping_reason="efficacy boundary crossed",
        total_observations=2000,
        relative_speedup_vs_fixed_horizon=0.35,
    )


@pytest.fixture
def boundary_result() -> BoundaryResult:
    return BoundaryResult(
        efficacy_boundaries=[2.96, 2.36, 1.96],
        futility_boundaries=[0.5, 1.0, 1.96],
        information_fractions=[0.33, 0.67, 1.0],
        alpha_spent=[0.005, 0.02, 0.05],
        adjusted_alpha=0.025,
    )


@pytest.fixture
def gs_result() -> GSResult:
    return GSResult(
        analysis_results=[{"look": 1, "pvalue": 0.12}],
        crossed_efficacy=False,
        crossed_futility=False,
        recommended_action="continue",
    )


@pytest.fixture
def bandit_result() -> BanditResult:
    return BanditResult(
        n_pulls_per_arm=[500, 300, 200],
        prob_best=[0.7, 0.2, 0.1],
        expected_loss=[0.001, 0.01, 0.03],
        current_best_arm=0,
        should_stop=False,
        total_reward=150.0,
        cumulative_regret=5.0,
        arm_means=[0.15, 0.12, 0.10],
        arm_credible_intervals=[(0.13, 0.17), (0.10, 0.14), (0.08, 0.12)],
    )


# ─── Test: creation with valid data ──────────────────────────────────


class TestCreation:
    def test_experiment_result(self, experiment_result: ExperimentResult) -> None:
        assert experiment_result.control_mean == 0.10
        assert experiment_result.significant is True
        assert experiment_result.method == "ztest"

    def test_sample_size_result(self, sample_size_result: SampleSizeResult) -> None:
        assert sample_size_result.n_per_variant == 5000
        assert sample_size_result.n_total == 10000
        assert sample_size_result.days_needed is None

    def test_srm_result(self, srm_result: SRMResult) -> None:
        assert srm_result.passed is False
        assert srm_result.worst_variant == 1

    def test_correction_result(self, correction_result: CorrectionResult) -> None:
        assert correction_result.n_rejected == 1
        assert correction_result.labels == ["metric_a", "metric_b", "metric_c"]

    def test_msprt_state(self, msprt_state: mSPRTState) -> None:
        assert msprt_state.should_stop is True
        assert msprt_state.mixture_lr == 5.2

    def test_msprt_result(self, msprt_result: mSPRTResult) -> None:
        assert msprt_result.stopping_reason == "efficacy boundary crossed"
        assert msprt_result.total_observations == 2000

    def test_boundary_result(self, boundary_result: BoundaryResult) -> None:
        assert len(boundary_result.efficacy_boundaries) == 3
        assert boundary_result.adjusted_alpha == 0.025

    def test_gs_result(self, gs_result: GSResult) -> None:
        assert gs_result.recommended_action == "continue"
        assert gs_result.crossed_efficacy is False

    def test_bandit_result(self, bandit_result: BanditResult) -> None:
        assert bandit_result.current_best_arm == 0
        assert len(bandit_result.arm_credible_intervals) == 3


# ─── Test: frozen dataclasses ────────────────────────────────────────


class TestFrozen:
    def test_experiment_result_frozen(self, experiment_result: ExperimentResult) -> None:
        with pytest.raises(AttributeError):
            experiment_result.pvalue = 0.99  # type: ignore[misc]

    def test_sample_size_result_frozen(self, sample_size_result: SampleSizeResult) -> None:
        with pytest.raises(AttributeError):
            sample_size_result.n_total = 999  # type: ignore[misc]

    def test_srm_result_frozen(self, srm_result: SRMResult) -> None:
        with pytest.raises(AttributeError):
            srm_result.passed = True  # type: ignore[misc]

    def test_correction_result_frozen(self, correction_result: CorrectionResult) -> None:
        with pytest.raises(AttributeError):
            correction_result.alpha = 0.01  # type: ignore[misc]

    def test_msprt_state_frozen(self, msprt_state: mSPRTState) -> None:
        with pytest.raises(AttributeError):
            msprt_state.should_stop = False  # type: ignore[misc]

    def test_msprt_result_frozen(self, msprt_result: mSPRTResult) -> None:
        with pytest.raises(AttributeError):
            msprt_result.stopping_reason = "nope"  # type: ignore[misc]

    def test_boundary_result_frozen(self, boundary_result: BoundaryResult) -> None:
        with pytest.raises(AttributeError):
            boundary_result.adjusted_alpha = 0.01  # type: ignore[misc]

    def test_gs_result_frozen(self, gs_result: GSResult) -> None:
        with pytest.raises(AttributeError):
            gs_result.recommended_action = "stop"  # type: ignore[misc]

    def test_bandit_result_frozen(self, bandit_result: BanditResult) -> None:
        with pytest.raises(AttributeError):
            bandit_result.current_best_arm = 2  # type: ignore[misc]


# ─── Test: to_dict returns plain Python types ────────────────────────


class TestToDict:
    def test_experiment_result_to_dict(self, experiment_result: ExperimentResult) -> None:
        d = experiment_result.to_dict()
        assert isinstance(d, dict)
        assert d["control_mean"] == 0.10
        assert d["significant"] is True
        assert isinstance(d["control_n"], int)

    def test_sample_size_result_to_dict(self, sample_size_result: SampleSizeResult) -> None:
        d = sample_size_result.to_dict()
        assert d["n_per_variant"] == 5000
        assert d["days_needed"] is None

    def test_srm_result_to_dict(self, srm_result: SRMResult) -> None:
        d = srm_result.to_dict()
        assert isinstance(d["observed"], list)
        assert all(isinstance(v, int) for v in d["observed"])

    def test_bandit_result_to_dict(self, bandit_result: BanditResult) -> None:
        d = bandit_result.to_dict()
        assert isinstance(d["arm_credible_intervals"], list)
        assert isinstance(d["arm_credible_intervals"][0], tuple)

    def test_to_dict_numpy_inside_dict(self) -> None:
        """Verify numpy values inside GSResult.analysis_results dicts are converted."""
        np = pytest.importorskip("numpy")
        result = GSResult(
            analysis_results=[{"look": np.int64(1), "pvalue": np.float64(0.12)}],
            crossed_efficacy=False,
            crossed_futility=False,
            recommended_action="continue",
        )
        d = result.to_dict()
        inner = d["analysis_results"][0]
        assert type(inner["look"]) is int
        assert type(inner["pvalue"]) is float

    def test_to_dict_with_numpy_types(self) -> None:
        """Verify numpy types are converted to plain Python."""
        np = pytest.importorskip("numpy")
        result = ExperimentResult(
            control_mean=np.float64(0.10),
            treatment_mean=np.float64(0.12),
            lift=np.float64(0.02),
            relative_lift=np.float64(0.20),
            pvalue=np.float64(0.0017),
            statistic=np.float64(3.14),
            ci_lower=np.float64(0.0075),
            ci_upper=np.float64(0.0325),
            significant=np.bool_(True),
            alpha=np.float64(0.05),
            method="ztest",
            metric="conversion",
            control_n=np.int64(5000),
            treatment_n=np.int64(5200),
            power=np.float64(0.92),
            effect_size=np.float64(0.064),
        )
        d = result.to_dict()
        assert type(d["control_mean"]) is float
        assert type(d["control_n"]) is int
        assert type(d["significant"]) is bool


# ─── Test: __repr__ smoke tests ──────────────────────────────────────


class TestRepr:
    def test_experiment_result_repr(self, experiment_result: ExperimentResult) -> None:
        r = repr(experiment_result)
        assert isinstance(r, str)
        assert "ExperimentResult" in r
        assert "conversion" in r
        assert "ztest" in r

    def test_sample_size_result_repr(self, sample_size_result: SampleSizeResult) -> None:
        r = repr(sample_size_result)
        assert isinstance(r, str)
        assert "SampleSizeResult" in r

    def test_srm_result_repr(self, srm_result: SRMResult) -> None:
        r = repr(srm_result)
        assert isinstance(r, str)
        assert "SRMResult" in r
        assert "mismatch" in r.lower()

    def test_srm_result_repr_passed(self) -> None:
        result = SRMResult(
            observed=[5000, 5000],
            expected_counts=[5000.0, 5000.0],
            chi2_statistic=0.0,
            pvalue=1.0,
            passed=True,
            alpha=0.01,
            deviations_pct=[0.0, 0.0],
            worst_variant=0,
            message="No sample ratio mismatch detected.",
        )
        r = repr(result)
        assert "No sample ratio mismatch" in r

    def test_correction_result_repr(self, correction_result: CorrectionResult) -> None:
        r = repr(correction_result)
        assert isinstance(r, str)
        assert "CorrectionResult" in r

    def test_msprt_state_repr(self, msprt_state: mSPRTState) -> None:
        r = repr(msprt_state)
        assert isinstance(r, str)
        assert "mSPRTState" in r

    def test_msprt_result_repr(self, msprt_result: mSPRTResult) -> None:
        r = repr(msprt_result)
        assert isinstance(r, str)
        assert "mSPRTResult" in r

    def test_boundary_result_repr(self, boundary_result: BoundaryResult) -> None:
        r = repr(boundary_result)
        assert isinstance(r, str)
        assert "BoundaryResult" in r

    def test_gs_result_repr(self, gs_result: GSResult) -> None:
        r = repr(gs_result)
        assert isinstance(r, str)
        assert "GSResult" in r

    def test_bandit_result_repr(self, bandit_result: BanditResult) -> None:
        r = repr(bandit_result)
        assert isinstance(r, str)
        assert "BanditResult" in r

    def test_experiment_result_repr_conversion(self, experiment_result: ExperimentResult) -> None:
        r = repr(experiment_result)
        assert "Cohen's h" in r

    def test_experiment_result_repr_continuous(self) -> None:
        result = ExperimentResult(
            control_mean=10.0,
            treatment_mean=12.0,
            lift=2.0,
            relative_lift=0.20,
            pvalue=0.03,
            statistic=2.1,
            ci_lower=0.5,
            ci_upper=3.5,
            significant=True,
            alpha=0.05,
            method="ttest",
            metric="continuous",
            control_n=100,
            treatment_n=100,
            power=0.80,
            effect_size=0.30,
        )
        r = repr(result)
        assert "Cohen's d" in r
        assert "Cohen's h" not in r


# ─── Test: SampleSizeResult.duration() ───────────────────────────────


class TestDuration:
    def test_duration_basic(self, sample_size_result: SampleSizeResult) -> None:
        result = sample_size_result.duration(daily_users=1000)
        assert result.days_needed == math.ceil(10000 / 1000)
        assert result.days_needed == 10
        # original unchanged
        assert sample_size_result.days_needed is None

    def test_duration_with_traffic_fraction(self, sample_size_result: SampleSizeResult) -> None:
        result = sample_size_result.duration(daily_users=1000, traffic_fraction=0.5)
        assert result.days_needed == math.ceil(10000 / (1000 * 0.5))
        assert result.days_needed == 20

    def test_duration_with_ramp_days(self, sample_size_result: SampleSizeResult) -> None:
        result = sample_size_result.duration(daily_users=1000, ramp_days=3)
        assert result.days_needed == math.ceil(10000 / 1000) + 3
        assert result.days_needed == 13

    def test_duration_with_all_params(self, sample_size_result: SampleSizeResult) -> None:
        result = sample_size_result.duration(
            daily_users=2000, traffic_fraction=0.25, ramp_days=5
        )
        expected = math.ceil(10000 / (2000 * 0.25)) + 5
        assert result.days_needed == expected
        assert result.days_needed == 25

    def test_duration_returns_new_instance(self, sample_size_result: SampleSizeResult) -> None:
        result = sample_size_result.duration(daily_users=1000)
        assert result is not sample_size_result
        # all other fields preserved
        assert result.n_per_variant == sample_size_result.n_per_variant
        assert result.n_total == sample_size_result.n_total
        assert result.alpha == sample_size_result.alpha
        assert result.metric == sample_size_result.metric

    def test_duration_repr_includes_days(self, sample_size_result: SampleSizeResult) -> None:
        result = sample_size_result.duration(daily_users=1000)
        r = repr(result)
        assert "days_needed" in r

    def test_duration_zero_daily_users_raises(self, sample_size_result: SampleSizeResult) -> None:
        with pytest.raises(ValueError, match="daily_users.*must be > 0"):
            sample_size_result.duration(daily_users=0)

    def test_duration_negative_traffic_fraction_raises(self, sample_size_result: SampleSizeResult) -> None:
        with pytest.raises(ValueError, match="traffic_fraction.*must be in"):
            sample_size_result.duration(daily_users=1000, traffic_fraction=-0.5)


# ─── Test: mSPRTResult has all mSPRTState fields ─────────────────────


class TestMSPRTFieldInheritance:
    def test_msprt_result_has_all_state_fields(self) -> None:
        from dataclasses import fields as dc_fields

        state_field_names = {f.name for f in dc_fields(mSPRTState)}
        result_field_names = {f.name for f in dc_fields(mSPRTResult)}
        assert state_field_names.issubset(result_field_names), (
            f"mSPRTResult is missing fields from mSPRTState: "
            f"{state_field_names - result_field_names}"
        )

    def test_msprt_result_has_extra_fields(self) -> None:
        from dataclasses import fields as dc_fields

        state_field_names = {f.name for f in dc_fields(mSPRTState)}
        result_field_names = {f.name for f in dc_fields(mSPRTResult)}
        extra = result_field_names - state_field_names
        assert "stopping_reason" in extra
        assert "total_observations" in extra
        assert "relative_speedup_vs_fixed_horizon" in extra
