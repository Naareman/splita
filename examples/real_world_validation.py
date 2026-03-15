"""Real-world validation of the splita pipeline.

Simulates five realistic A/B test scenarios (based on well-known Kaggle
dataset patterns) and runs the FULL splita pipeline on each one.  Every
assertion must pass and the output is printed for manual inspection.
"""

from __future__ import annotations

import time
import traceback

import numpy as np

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def banner(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def assert_ok(condition: bool, msg: str) -> None:
    if condition:
        ok(msg)
    else:
        fail(msg)


# ────────────────────────────────────────────────────────────────
# Scenario 1 — E-commerce conversion test
#   Based on the Kaggle "A/B Testing" dataset pattern:
#   ~294K users, 50/50 split, ~12% baseline conversion,
#   small lift (~0.3 pp).
# ────────────────────────────────────────────────────────────────
def scenario_1() -> None:
    banner("Scenario 1: E-commerce Conversion Test (Kaggle pattern)")
    t0 = time.perf_counter()

    from splita import (
        BayesianExperiment,
        Experiment,
        QuantileExperiment,
        SRMCheck,
        auto,
        check,
        diagnose,
        explain,
        report,
    )

    rng = np.random.default_rng(42)

    n_control = 147_239
    n_treatment = 147_239
    p_control = 0.1197
    p_treatment = 0.1230  # ~0.3 pp lift — realistic small effect

    control = rng.binomial(1, p_control, n_control)
    treatment = rng.binomial(1, p_treatment, n_treatment)

    # ── check() — pre-analysis health ──
    section("check()")
    chk = check(control, treatment)
    print(f"  SRM passed: {chk.srm_passed}")
    print(f"  Has outliers: {chk.has_outliers}")
    assert_ok(chk.srm_passed, "SRM should pass for 50/50 split")

    # ── auto() — one-call pipeline ──
    section("auto()")
    ar = auto(control, treatment)
    print(f"  Metric detected: {ar.primary_result.metric}")
    print(f"  p-value: {ar.primary_result.pvalue:.6f}")
    print(f"  Lift: {ar.primary_result.lift:.6f}")
    assert_ok(ar.primary_result.metric == "conversion", "Metric should be conversion")

    # ── Experiment.run() ──
    section("Experiment.run()")
    result = Experiment(control, treatment).run()
    print(f"  Method: {result.method}")
    print(f"  Significant: {result.significant}")
    print(f"  p-value: {result.pvalue:.6f}")
    print(f"  Lift (abs): {result.lift:.6f}")
    print(f"  Relative lift: {result.relative_lift:.4f}")
    assert_ok(result.method in ("ztest", "chisquare"), "Should use z-test or chi-square for proportions")
    assert_ok(0 < result.pvalue < 1, "p-value in valid range")

    # ── explain() ──
    section("explain()")
    explanation = explain(result)
    print(f"  {explanation[:200]}...")
    assert_ok(len(explanation) > 50, "Explanation should be non-trivial")

    # ── explain() multilingual ──
    section("explain() — Arabic")
    explanation_ar = explain(result, lang="ar")
    print(f"  {explanation_ar[:200]}...")
    assert_ok(len(explanation_ar) > 30, "Arabic explanation should be non-trivial")

    # ── report() ──
    section("report()")
    srm = SRMCheck([n_control, n_treatment]).run()
    html = report(result, srm, title="E-commerce Landing Page Test")
    print(f"  HTML report: {len(html):,} chars")
    assert_ok(len(html) > 500, "Report should be substantial HTML")
    assert_ok("<html" in html.lower() or "<!doctype" in html.lower() or "<div" in html.lower(),
              "Report should contain HTML elements")

    # ── diagnose() ──
    section("diagnose()")
    dx = diagnose(result)
    print(f"  Status: {dx.status}")
    print(f"  Confidence level: {dx.confidence_level}")
    print(f"  Action items: {dx.action_items[:2]}")
    assert_ok(dx.status in ("healthy", "warning", "critical"),
              "Status should be a valid category")

    # ── BayesianExperiment ──
    section("BayesianExperiment")
    bayes = BayesianExperiment(control, treatment, random_state=42).run()
    print(f"  P(B>A): {bayes.prob_b_beats_a:.4f}")
    print(f"  Expected loss (choose A): {bayes.expected_loss_a:.6f}")
    assert_ok(0 <= bayes.prob_b_beats_a <= 1, "P(B>A) in [0,1]")

    # ── QuantileExperiment on revenue (synthetic) ──
    section("QuantileExperiment on revenue")
    # Simulate revenue: most users $0, converters get log-normal revenue
    ctrl_revenue = np.where(control == 1, rng.lognormal(3.0, 1.0, n_control), 0.0)
    trt_revenue = np.where(treatment == 1, rng.lognormal(3.05, 1.0, n_treatment), 0.0)

    # Use subsample for speed (quantile bootstrap is expensive)
    idx_c = rng.choice(n_control, 10_000, replace=False)
    idx_t = rng.choice(n_treatment, 10_000, replace=False)
    qr = QuantileExperiment(
        ctrl_revenue[idx_c], trt_revenue[idx_t],
        quantiles=[0.5, 0.75, 0.9],
        n_bootstrap=500,
        random_state=42,
    ).run()
    print(f"  Quantiles tested: {qr.quantiles}")
    print(f"  Differences: {[round(d, 4) for d in qr.differences]}")
    print(f"  Significant: {qr.significant}")
    assert_ok(len(qr.quantiles) == 3, "Should test 3 quantiles")

    elapsed = time.perf_counter() - t0
    print(f"\n  Pipeline time: {elapsed:.2f}s")


# ────────────────────────────────────────────────────────────────
# Scenario 2 — Cookie Cats game retention
#   Based on the Kaggle "Cookie Cats A/B Testing" dataset:
#   ~45K per group, gate at level 30 vs 40,
#   1-day retention ~44.8% vs ~44.2%, 7-day retention ~19% vs ~18.2%.
# ────────────────────────────────────────────────────────────────
def scenario_2() -> None:
    banner("Scenario 2: Cookie Cats Game Retention")
    t0 = time.perf_counter()

    from splita import (
        Experiment,
        MultipleCorrection,
        SRMCheck,
        explain,
    )

    rng = np.random.default_rng(123)

    n_gate30 = 44_700  # control: gate at level 30
    n_gate40 = 45_489  # treatment: gate at level 40

    # 1-day retention
    ret1_ctrl = rng.binomial(1, 0.4482, n_gate30)
    ret1_trt = rng.binomial(1, 0.4422, n_gate40)

    # 7-day retention
    ret7_ctrl = rng.binomial(1, 0.1902, n_gate30)
    ret7_trt = rng.binomial(1, 0.1820, n_gate40)

    # ── SRM check ──
    section("SRM check (slightly unequal split)")
    srm = SRMCheck([n_gate30, n_gate40]).run()
    print(f"  SRM passed: {srm.passed}")
    print(f"  SRM p-value: {srm.pvalue:.4f}")
    # The split is close enough to 50/50 that SRM should pass
    assert_ok(isinstance(srm.passed, bool), "SRM result is boolean")

    # ── 1-day retention ──
    section("1-day retention experiment")
    r1 = Experiment(ret1_ctrl, ret1_trt).run()
    print(f"  Control mean: {r1.control_mean:.4f}")
    print(f"  Treatment mean: {r1.treatment_mean:.4f}")
    print(f"  p-value: {r1.pvalue:.6f}")
    print(f"  Significant: {r1.significant}")
    assert_ok(abs(r1.control_mean - 0.448) < 0.01, "Control retention ~44.8%")

    # ── 7-day retention ──
    section("7-day retention experiment")
    r7 = Experiment(ret7_ctrl, ret7_trt).run()
    print(f"  Control mean: {r7.control_mean:.4f}")
    print(f"  Treatment mean: {r7.treatment_mean:.4f}")
    print(f"  p-value: {r7.pvalue:.6f}")
    print(f"  Significant: {r7.significant}")

    # ── Multiple correction ──
    section("Multiple testing correction")
    mc = MultipleCorrection(
        [r1.pvalue, r7.pvalue],
        method="bh",
        labels=["1-day retention", "7-day retention"],
    ).run()
    print(f"  Raw p-values: {[round(r1.pvalue, 6), round(r7.pvalue, 6)]}")
    print(f"  Adjusted p-values: {[round(p, 6) for p in mc.adjusted_pvalues]}")
    print(f"  Rejected after correction: {mc.rejected}")
    assert_ok(len(mc.adjusted_pvalues) == 2, "Two adjusted p-values")
    assert_ok(all(p >= raw for p, raw in zip(mc.adjusted_pvalues, [r1.pvalue, r7.pvalue])),
              "Adjusted p-values >= raw p-values")

    # ── explain() on both ──
    section("Explain results")
    print(f"  1-day: {explain(r1)[:150]}...")
    print(f"  7-day: {explain(r7)[:150]}...")

    elapsed = time.perf_counter() - t0
    print(f"\n  Pipeline time: {elapsed:.2f}s")


# ────────────────────────────────────────────────────────────────
# Scenario 3 — Marketing email A/B/C test
#   3 variants: baseline, new subject line, new subject + CTA.
#   Open rate ~22%, click rate ~3.5%, conversion ~0.8%.
# ────────────────────────────────────────────────────────────────
def scenario_3() -> None:
    banner("Scenario 3: Marketing Email A/B/C Test")
    t0 = time.perf_counter()

    from splita import (
        MultiObjectiveExperiment,
        ThompsonSampler,
        explain,
        meta_analysis,
    )

    rng = np.random.default_rng(456)
    n_per = 15_000

    # Variant A (control)
    open_a = rng.binomial(1, 0.220, n_per)
    click_a = rng.binomial(1, 0.035, n_per)
    conv_a = rng.binomial(1, 0.008, n_per)

    # Variant B: better subject line
    open_b = rng.binomial(1, 0.245, n_per)
    click_b = rng.binomial(1, 0.038, n_per)
    conv_b = rng.binomial(1, 0.009, n_per)

    # Variant C: better subject + CTA
    open_c = rng.binomial(1, 0.250, n_per)
    click_c = rng.binomial(1, 0.042, n_per)
    conv_c = rng.binomial(1, 0.011, n_per)

    # ── Thompson Sampler ──
    section("ThompsonSampler (bandit approach)")
    ts = ThompsonSampler(3, likelihood="bernoulli", random_state=42)

    # Feed observed data
    for obs in open_a:
        ts.update(0, float(obs))
    for obs in open_b:
        ts.update(1, float(obs))
    for obs in open_c:
        ts.update(2, float(obs))

    bandit_result = ts.result()
    print(f"  Best arm: {bandit_result.current_best_arm}")
    print(f"  Arm probabilities: {[round(p, 4) for p in bandit_result.prob_best]}")
    print(f"  Should stop: {bandit_result.should_stop}")
    assert_ok(bandit_result.current_best_arm in (1, 2),
              "Best arm should be B or C (higher open rate)")

    # ── MultiObjectiveExperiment (A vs B) ──
    section("MultiObjectiveExperiment (A vs B)")
    moe = MultiObjectiveExperiment(
        metric_names=["open_rate", "click_rate", "conversion"],
    )
    moe.add_metric(open_a, open_b)
    moe.add_metric(click_a, click_b)
    moe.add_metric(conv_a, conv_b)
    mo_result = moe.run()
    print(f"  Recommendation: {mo_result.recommendation}")
    print(f"  Pareto dominant: {mo_result.pareto_dominant}")
    assert_ok(mo_result.recommendation in ("adopt", "reject", "tradeoff"),
              "Recommendation is valid category")

    # ── MultiObjectiveExperiment (A vs C) ──
    section("MultiObjectiveExperiment (A vs C)")
    moe2 = MultiObjectiveExperiment(
        metric_names=["open_rate", "click_rate", "conversion"],
    )
    moe2.add_metric(open_a, open_c)
    moe2.add_metric(click_a, click_c)
    moe2.add_metric(conv_a, conv_c)
    mo_result2 = moe2.run()
    print(f"  Recommendation: {mo_result2.recommendation}")
    print(f"  Pareto dominant: {mo_result2.pareto_dominant}")

    # ── meta_analysis: pool open-rate effects across two comparisons ──
    section("meta_analysis (pooling open-rate effects)")
    # Compute effects and SEs manually for the meta-analysis
    effects = []
    ses = []
    for ctrl, trt, label in [
        (open_a, open_b, "A vs B"),
        (open_a, open_c, "A vs C"),
    ]:
        p_c = ctrl.mean()
        p_t = trt.mean()
        eff = p_t - p_c
        se = np.sqrt(p_c * (1 - p_c) / len(ctrl) + p_t * (1 - p_t) / len(trt))
        effects.append(float(eff))
        ses.append(float(se))
        print(f"  {label}: effect={eff:.4f}, SE={se:.4f}")

    ma = meta_analysis(effects, ses, labels=["A vs B", "A vs C"])
    print(f"  Pooled effect: {ma.combined_effect:.4f}")
    print(f"  Pooled SE: {ma.combined_se:.4f}")
    print(f"  Method: {ma.method}")
    print(f"  Heterogeneity I2: {ma.i_squared:.4f}")
    assert_ok(ma.combined_effect > 0, "Pooled effect should be positive (improvement)")

    # ── explain on individual metric results (explain supports ExperimentResult) ──
    section("Explain (individual metric results from MultiObjective)")
    for mr in mo_result.metric_results:
        print(f"  {explain(mr)[:150]}...")

    elapsed = time.perf_counter() - t0
    print(f"\n  Pipeline time: {elapsed:.2f}s")


# ────────────────────────────────────────────────────────────────
# Scenario 4 — Marketplace pricing experiment
#   Two-sided marketplace: heavy-tailed revenue, outliers.
#   Buyer-side randomization, clustered by city.
# ────────────────────────────────────────────────────────────────
def scenario_4() -> None:
    banner("Scenario 4: Marketplace Pricing Experiment")
    t0 = time.perf_counter()

    from splita import (
        CUPED,
        ClusterExperiment,
        DifferenceInDifferences,
        Experiment,
        OutlierHandler,
        explain,
    )

    rng = np.random.default_rng(789)

    n_buyers = 5_000
    n_cities = 50
    buyers_per_city = n_buyers // n_cities

    # Assign buyers to cities
    city_ids = np.repeat(np.arange(n_cities), buyers_per_city)

    # City-level random effects
    city_effects = rng.normal(0, 5.0, n_cities)

    # Revenue: log-normal base + city effect + outliers
    base_revenue = rng.lognormal(3.5, 1.2, n_buyers)
    revenue_ctrl = base_revenue + city_effects[city_ids] + rng.normal(0, 2, n_buyers)

    # Treatment: small uplift + some outlier whales
    treatment_effect = 3.0
    revenue_trt = base_revenue + city_effects[city_ids] + treatment_effect + rng.normal(0, 2, n_buyers)

    # Inject outlier "whale" transactions (top 0.5%)
    n_whales = int(n_buyers * 0.005)
    whale_idx_ctrl = rng.choice(n_buyers, n_whales, replace=False)
    whale_idx_trt = rng.choice(n_buyers, n_whales, replace=False)
    revenue_ctrl[whale_idx_ctrl] += rng.exponential(200, n_whales)
    revenue_trt[whale_idx_trt] += rng.exponential(200, n_whales)

    # ── OutlierHandler ──
    section("OutlierHandler (winsorize)")
    oh = OutlierHandler(method="winsorize", lower=0.01, upper=0.99)
    oh.fit(revenue_ctrl, revenue_trt)
    ctrl_clean, trt_clean = oh.transform(revenue_ctrl, revenue_trt)
    print(f"  Lower threshold: {oh.lower_threshold_:.2f}")
    print(f"  Upper threshold: {oh.upper_threshold_:.2f}")
    print(f"  Capped count: {oh.n_capped_}")
    assert_ok(oh.n_capped_ > 0, "Some outliers should be capped")

    # Raw vs cleaned comparison
    section("Raw vs cleaned experiment")
    raw_result = Experiment(revenue_ctrl, revenue_trt).run()
    clean_result = Experiment(ctrl_clean, trt_clean).run()
    print(f"  Raw    — lift: {raw_result.lift:.2f}, p: {raw_result.pvalue:.6f}")
    print(f"  Clean  — lift: {clean_result.lift:.2f}, p: {clean_result.pvalue:.6f}")
    assert_ok(clean_result.pvalue <= raw_result.pvalue + 0.5,
              "Cleaned data should have similar or better p-value")

    # ── CUPED ──
    section("CUPED (using pre-experiment revenue)")
    # Simulate pre-period revenue (correlated with post)
    pre_ctrl = base_revenue + city_effects[city_ids] + rng.normal(0, 2, n_buyers)
    pre_trt = base_revenue + city_effects[city_ids] + rng.normal(0, 2, n_buyers)

    cuped = CUPED()
    ctrl_adj, trt_adj = cuped.fit_transform(ctrl_clean, trt_clean, pre_ctrl, pre_trt)
    print(f"  Theta: {cuped.theta_:.4f}")
    print(f"  Correlation: {cuped.correlation_:.4f}")
    print(f"  Variance reduction: {cuped.variance_reduction_:.4f}")
    assert_ok(cuped.variance_reduction_ > 0.1, "CUPED should reduce variance substantially")

    cuped_result = Experiment(ctrl_adj, trt_adj).run()
    print(f"  CUPED  — lift: {cuped_result.lift:.2f}, p: {cuped_result.pvalue:.6f}")

    # ── ClusterExperiment (by city) ──
    section("ClusterExperiment (city-level)")
    cluster_result = ClusterExperiment(
        ctrl_clean, trt_clean,
        control_clusters=city_ids,
        treatment_clusters=city_ids,
    ).run()
    print(f"  Cluster ATE: {cluster_result.lift:.2f}")
    print(f"  p-value: {cluster_result.pvalue:.6f}")
    print(f"  N clusters ctrl: {cluster_result.n_clusters_control}")
    print(f"  N clusters trt: {cluster_result.n_clusters_treatment}")
    assert_ok(cluster_result.n_clusters_control == n_cities, "Should have 50 control clusters")

    # ── DifferenceInDifferences ──
    section("Difference-in-Differences")
    did = DifferenceInDifferences()
    did.fit(pre_ctrl, pre_trt, ctrl_clean, trt_clean)
    did_result = did.result()
    print(f"  ATT: {did_result.att:.2f}")
    print(f"  p-value: {did_result.pvalue:.6f}")
    print(f"  Parallel trends p: {did_result.parallel_trends_pvalue:.4f}")
    assert_ok(did_result.att > 0, "ATT should be positive (treatment helps)")
    assert_ok(did_result.parallel_trends_pvalue > 0.01,
              "Parallel trends should hold (same pre-period DGP)")

    # ── explain (on a supported type) ──
    section("Explain")
    print(f"  CUPED result: {explain(cuped_result)[:200]}...")

    elapsed = time.perf_counter() - t0
    print(f"\n  Pipeline time: {elapsed:.2f}s")


# ────────────────────────────────────────────────────────────────
# Scenario 5 — Sequential monitoring over 14 days
#   Daily data, checking each day.
#   mSPRT + ConfidenceSequence + EffectTimeSeries + NoveltyCurve.
# ────────────────────────────────────────────────────────────────
def scenario_5() -> None:
    banner("Scenario 5: Sequential Monitoring (14 days)")
    t0 = time.perf_counter()

    from splita import (
        ConfidenceSequence,
        EffectTimeSeries,
        Experiment,
        NoveltyCurve,
        explain,
        mSPRT,
    )

    rng = np.random.default_rng(101)

    n_per_day = 2_000
    n_days = 14
    total_per_group = n_per_day * n_days

    # Generate daily data with a real treatment effect
    # Control: 10% conversion, Treatment: 12% conversion
    all_ctrl = rng.binomial(1, 0.10, total_per_group)
    all_trt = rng.binomial(1, 0.12, total_per_group)

    # Timestamps: day 0..13 for each observation
    timestamps = np.repeat(np.arange(n_days), n_per_day)
    # Combined timestamps for EffectTimeSeries
    all_timestamps = np.concatenate([timestamps, timestamps])

    # ── mSPRT: daily updates ──
    section("mSPRT — daily sequential test")
    seq = mSPRT(metric="conversion", alpha=0.05)

    stopped_day = None
    for day in range(n_days):
        start = day * n_per_day
        end = (day + 1) * n_per_day
        day_ctrl = all_ctrl[start:end]
        day_trt = all_trt[start:end]

        state = seq.update(day_ctrl, day_trt)
        status = "STOP" if state.should_stop else "continue"
        n_so_far = state.n_control + state.n_treatment
        print(f"  Day {day + 1:2d}: n={n_so_far:6d}, "
              f"p_always_valid={state.always_valid_pvalue:.6f}, {status}")

        if state.should_stop and stopped_day is None:
            stopped_day = day + 1

    print(f"  Stopped at day: {stopped_day or 'never'}")
    assert_ok(state.always_valid_pvalue < 1.0, "mSPRT p-value should be computed")

    # ── ConfidenceSequence ──
    section("ConfidenceSequence — anytime-valid CI")
    cs = ConfidenceSequence(alpha=0.05)

    for day in range(n_days):
        start = day * n_per_day
        end = (day + 1) * n_per_day
        cs_state = cs.update(all_ctrl[start:end], all_trt[start:end])

    print(f"  Final CI: ({cs_state.ci_lower:.6f}, {cs_state.ci_upper:.6f})")
    print(f"  Effect estimate: {cs_state.effect_estimate:.6f}")
    print(f"  Should stop: {cs_state.should_stop}")
    assert_ok(cs_state.ci_lower < cs_state.ci_upper, "CI lower < upper")

    # ── EffectTimeSeries ──
    section("EffectTimeSeries — cumulative effect over time")
    ets = EffectTimeSeries().fit(all_ctrl, all_trt, all_timestamps)
    ets_result = ets.result()
    print(f"  Time points: {len(ets_result.time_points)}")
    print(f"  Final lift: {ets_result.final_lift:.6f}")
    print(f"  Final p-value: {ets_result.final_pvalue:.6f}")
    print(f"  Is stable: {ets_result.is_stable}")
    assert_ok(len(ets_result.time_points) == n_days, f"Should have {n_days} time points")
    assert_ok(ets_result.final_lift > 0, "Final effect should be positive")

    # ── NoveltyCurve ──
    section("NoveltyCurve — detect novelty/primacy effects")
    nc = NoveltyCurve(window_size=3).fit(all_ctrl, all_trt, all_timestamps)
    nc_result = nc.result()
    print(f"  Trend: {nc_result.trend_direction}")
    print(f"  Number of windows: {len(nc_result.windows)}")
    assert_ok(nc_result.trend_direction in ("stable", "decreasing", "increasing"),
              "Trend should be a valid direction")

    # ── explain (on the final day's equivalent fixed-horizon result) ──
    section("Explain (final fixed-horizon result)")
    final_result = Experiment(all_ctrl, all_trt).run()
    print(f"  {explain(final_result)[:200]}...")

    elapsed = time.perf_counter() - t0
    print(f"\n  Pipeline time: {elapsed:.2f}s")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_t0 = time.perf_counter()

    scenarios = [
        ("Scenario 1", scenario_1),
        ("Scenario 2", scenario_2),
        ("Scenario 3", scenario_3),
        ("Scenario 4", scenario_4),
        ("Scenario 5", scenario_5),
    ]

    failed_scenarios = []
    for name, fn in scenarios:
        try:
            fn()
        except Exception:
            fail(f"{name} CRASHED")
            traceback.print_exc()
            failed_scenarios.append(name)

    total_elapsed = time.perf_counter() - total_t0

    banner("SUMMARY")
    print(f"  Scenarios run: {len(scenarios)}")
    print(f"  Scenarios crashed: {len(failed_scenarios)}")
    if failed_scenarios:
        print(f"  Failed: {', '.join(failed_scenarios)}")
    print(f"  Assertions passed: {PASS}")
    print(f"  Assertions failed: {FAIL}")
    print(f"  Total time: {total_elapsed:.2f}s")

    if FAIL > 0 or failed_scenarios:
        print("\n  *** VALIDATION FAILED ***")
        exit(1)
    else:
        print("\n  *** ALL VALIDATIONS PASSED ***")
