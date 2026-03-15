"""How to use splita with real Kaggle A/B test CSV datasets.

Shows the pandas integration patterns a user would actually follow when
loading data from Kaggle and feeding it into splita.  Each example
generates a DataFrame that mimics the exact schema of a well-known Kaggle
dataset, then demonstrates the idiomatic splita workflow.
"""

from __future__ import annotations

import numpy as np

# We generate data in-process so the example runs without downloads.
# The column names and dtypes match the real Kaggle datasets exactly.
try:
    import pandas as pd
except ImportError:
    raise SystemExit(
        "This example requires pandas.  Install with: pip install pandas"
    )


# ────────────────────────────────────────────────────────────────
# Pattern 1 — Kaggle "A/B Testing" dataset
#   Columns: user_id, timestamp, group, landing_page, converted
#   https://www.kaggle.com/datasets/zhangluyuan/ab-testing
# ────────────────────────────────────────────────────────────────
def kaggle_ab_testing_pattern() -> None:
    print("\n" + "=" * 60)
    print("  Pattern 1: Kaggle 'A/B Testing' Dataset")
    print("=" * 60)

    from splita import Experiment, SRMCheck, auto, check, explain, report

    rng = np.random.default_rng(42)
    n = 294_478

    # Simulate the CSV structure
    df = pd.DataFrame({
        "user_id": np.arange(n),
        "timestamp": pd.date_range("2017-01-02", periods=n, freq="2min"),
        "group": rng.choice(["control", "treatment"], n),
        "landing_page": "",  # filled below
        "converted": 0,  # filled below
    })
    df.loc[df["group"] == "control", "landing_page"] = "old_page"
    df.loc[df["group"] == "treatment", "landing_page"] = "new_page"
    df.loc[df["group"] == "control", "converted"] = rng.binomial(
        1, 0.1197, (df["group"] == "control").sum()
    )
    df.loc[df["group"] == "treatment", "converted"] = rng.binomial(
        1, 0.1230, (df["group"] == "treatment").sum()
    )

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    # ── The idiomatic splita workflow ──

    # Step 1: Extract arrays from DataFrame
    control = df.loc[df["group"] == "control", "converted"].values
    treatment = df.loc[df["group"] == "treatment", "converted"].values
    print(f"\nControl n={len(control):,}, Treatment n={len(treatment):,}")

    # Step 2: Pre-analysis checks
    health = check(control, treatment)
    print(f"\nSRM passed: {health.srm_passed}")

    # Step 3: One-line analysis
    result = auto(control, treatment)
    print(f"\n{explain(result.primary_result)}")

    # Step 4: Full experiment with more control
    exp = Experiment(control, treatment).run()
    print(f"\nDetailed result:")
    print(f"  Control rate: {exp.control_mean:.4f}")
    print(f"  Treatment rate: {exp.treatment_mean:.4f}")
    print(f"  Lift: {exp.lift:+.4f}")
    print(f"  p-value: {exp.pvalue:.6f}")
    print(f"  Significant: {exp.significant}")

    # Step 5: Generate report
    srm = SRMCheck([len(control), len(treatment)]).run()
    html = report(exp, srm, title="Landing Page A/B Test")
    print(f"\nGenerated {len(html):,} char HTML report")


# ────────────────────────────────────────────────────────────────
# Pattern 2 — Kaggle "Cookie Cats" dataset
#   Columns: userid, version, sum_gamerounds, retention_1, retention_7
#   https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing
# ────────────────────────────────────────────────────────────────
def kaggle_cookie_cats_pattern() -> None:
    print("\n" + "=" * 60)
    print("  Pattern 2: Kaggle 'Cookie Cats' Dataset")
    print("=" * 60)

    from splita import Experiment, MultipleCorrection, explain

    rng = np.random.default_rng(123)

    # Simulate the CSV structure
    n = 90_189
    versions = np.array(["gate_30"] * 44_700 + ["gate_40"] * 45_489)
    rng.shuffle(versions)

    df = pd.DataFrame({
        "userid": np.arange(1, n + 1),
        "version": versions,
        "sum_gamerounds": rng.negative_binomial(5, 0.05, n),
        "retention_1": False,
        "retention_7": False,
    })

    # Fill retention based on version
    mask_30 = df["version"] == "gate_30"
    mask_40 = df["version"] == "gate_40"
    df.loc[mask_30, "retention_1"] = rng.random(mask_30.sum()) < 0.4482
    df.loc[mask_40, "retention_1"] = rng.random(mask_40.sum()) < 0.4422
    df.loc[mask_30, "retention_7"] = rng.random(mask_30.sum()) < 0.1902
    df.loc[mask_40, "retention_7"] = rng.random(mask_40.sum()) < 0.1820

    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())
    print(f"\nRetention rates by version:")
    print(df.groupby("version")[["retention_1", "retention_7"]].mean())

    # ── Using groupby to extract data ──
    gate_30 = df[df["version"] == "gate_30"]
    gate_40 = df[df["version"] == "gate_40"]

    # Test both metrics
    results = {}
    for metric in ["retention_1", "retention_7"]:
        ctrl = gate_30[metric].astype(int).values
        trt = gate_40[metric].astype(int).values
        r = Experiment(ctrl, trt).run()
        results[metric] = r
        print(f"\n{metric}:")
        print(f"  {explain(r)[:200]}")

    # Apply multiple testing correction
    pvals = [results[m].pvalue for m in results]
    mc = MultipleCorrection(
        pvals,
        method="holm",
        labels=list(results.keys()),
    ).run()
    print(f"\nMultiple correction (Holm):")
    print(f"  Raw p-values: {[round(p, 4) for p in pvals]}")
    print(f"  Adjusted: {[round(p, 4) for p in mc.adjusted_pvalues]}")
    print(f"  Rejected: {mc.rejected}")


# ────────────────────────────────────────────────────────────────
# Pattern 3 — E-commerce revenue with pandas operations
#   Typical revenue dataset with user_id, variant, sessions,
#   revenue, country.
# ────────────────────────────────────────────────────────────────
def ecommerce_revenue_pattern() -> None:
    print("\n" + "=" * 60)
    print("  Pattern 3: E-commerce Revenue (pandas workflow)")
    print("=" * 60)

    from splita import (
        CUPED,
        BayesianExperiment,
        Experiment,
        OutlierHandler,
        QuantileExperiment,
        explain,
    )

    rng = np.random.default_rng(789)
    n = 20_000

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "variant": rng.choice(["control", "treatment"], n),
        "pre_revenue": rng.lognormal(2.5, 1.5, n),
        "country": rng.choice(["US", "UK", "DE", "FR", "JP"], n, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
    })

    # Post-revenue correlated with pre-revenue + treatment effect
    treatment_mask = df["variant"] == "treatment"
    noise = rng.normal(0, 5, n)
    df["revenue"] = df["pre_revenue"] * 0.8 + noise + 2.0
    df.loc[treatment_mask, "revenue"] += 1.5  # treatment uplift

    # Add some whale outliers
    whale_idx = rng.choice(n, 50, replace=False)
    df.loc[whale_idx, "revenue"] += rng.exponential(100, 50)

    print(f"\nDataFrame shape: {df.shape}")
    print(df.describe())

    ctrl_mask = df["variant"] == "control"
    trt_mask = df["variant"] == "treatment"

    ctrl_rev = df.loc[ctrl_mask, "revenue"].values
    trt_rev = df.loc[trt_mask, "revenue"].values

    # ── Step 1: Handle outliers ──
    print("\n--- Outlier Handling ---")
    oh = OutlierHandler(method="winsorize")
    oh.fit(ctrl_rev, trt_rev)
    ctrl_clean, trt_clean = oh.transform(ctrl_rev, trt_rev)
    print(f"Capped {oh.n_capped_} outlier values")

    # ── Step 2: CUPED with pre-revenue ──
    print("\n--- CUPED Variance Reduction ---")
    ctrl_pre = df.loc[ctrl_mask, "pre_revenue"].values
    trt_pre = df.loc[trt_mask, "pre_revenue"].values
    cuped = CUPED()
    ctrl_adj, trt_adj = cuped.fit_transform(ctrl_clean, trt_clean, ctrl_pre, trt_pre)
    print(f"Variance reduction: {cuped.variance_reduction_:.1%}")

    # ── Step 3: Run experiment ──
    print("\n--- Frequentist Result ---")
    result = Experiment(ctrl_adj, trt_adj).run()
    print(explain(result))

    # ── Step 4: Bayesian for business decision ──
    print("\n--- Bayesian Result ---")
    bayes = BayesianExperiment(ctrl_adj, trt_adj, random_state=42).run()
    print(f"P(treatment > control): {bayes.prob_b_beats_a:.4f}")
    print(f"Expected loss if we choose A: {bayes.expected_loss_a:.4f}")

    # ── Step 5: Quantile analysis ──
    print("\n--- Quantile Analysis (median, p75, p90) ---")
    qr = QuantileExperiment(
        ctrl_clean, trt_clean,
        quantiles=[0.5, 0.75, 0.9],
        n_bootstrap=1000,
        random_state=42,
    ).run()
    for q, diff, sig in zip(qr.quantiles, qr.differences, qr.significant):
        print(f"  Q{q:.0%}: diff={diff:+.2f}, significant={sig}")

    # ── Step 6: Segment analysis with groupby ──
    print("\n--- Per-Country Analysis ---")
    for country in ["US", "UK"]:
        c_mask = ctrl_mask & (df["country"] == country)
        t_mask = trt_mask & (df["country"] == country)
        if c_mask.sum() < 100 or t_mask.sum() < 100:
            continue
        r = Experiment(
            df.loc[c_mask, "revenue"].values,
            df.loc[t_mask, "revenue"].values,
        ).run()
        print(f"  {country}: lift={r.lift:+.2f}, p={r.pvalue:.4f}, sig={r.significant}")


# ────────────────────────────────────────────────────────────────
# Pattern 4 — Time-series format with daily aggregation
#   Common in dashboards: date, variant, metric columns.
# ────────────────────────────────────────────────────────────────
def timeseries_dashboard_pattern() -> None:
    print("\n" + "=" * 60)
    print("  Pattern 4: Time-series Dashboard Format")
    print("=" * 60)

    from splita import EffectTimeSeries, Experiment, mSPRT, explain

    rng = np.random.default_rng(999)

    # Daily aggregated data (common dashboard export format)
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    daily_n = 3_000

    rows = []
    for date in dates:
        for variant in ["control", "treatment"]:
            p = 0.10 if variant == "control" else 0.12
            conversions = rng.binomial(daily_n, p)
            rows.append({
                "date": date,
                "variant": variant,
                "users": daily_n,
                "conversions": conversions,
                "conversion_rate": conversions / daily_n,
            })

    df = pd.DataFrame(rows)
    print(f"\nDaily summary table:")
    print(df.head(6))

    # ── Expand to user-level for splita ──
    print("\n--- Expanding to user-level data ---")
    all_ctrl = []
    all_trt = []
    ctrl_timestamps = []
    trt_timestamps = []

    for _, row in df.iterrows():
        day_idx = (row["date"] - dates[0]).days
        n_conv = int(row["conversions"])
        n_total = int(row["users"])
        data = np.zeros(n_total)
        data[:n_conv] = 1
        rng.shuffle(data)

        if row["variant"] == "control":
            all_ctrl.append(data)
            ctrl_timestamps.append(np.full(n_total, day_idx))
        else:
            all_trt.append(data)
            trt_timestamps.append(np.full(n_total, day_idx))

    control = np.concatenate(all_ctrl)
    treatment = np.concatenate(all_trt)
    ts_ctrl = np.concatenate(ctrl_timestamps)
    ts_trt = np.concatenate(trt_timestamps)
    all_ts = np.concatenate([ts_ctrl, ts_trt])

    print(f"Total control: {len(control):,}, treatment: {len(treatment):,}")

    # ── Fixed-horizon test ──
    print("\n--- Final fixed-horizon test ---")
    result = Experiment(control, treatment).run()
    print(explain(result))

    # ── Sequential monitoring ──
    print("\n--- Sequential mSPRT monitoring ---")
    seq = mSPRT(metric="conversion")
    for day in range(14):
        c = control[ts_ctrl == day]
        t = treatment[ts_trt == day]
        state = seq.update(c, t)
        if state.should_stop:
            print(f"  Stopped at day {day + 1} (p={state.always_valid_pvalue:.6f})")
            break
    else:
        print(f"  Did not stop after 14 days (p={state.always_valid_pvalue:.6f})")

    # ── EffectTimeSeries ──
    print("\n--- Effect over time ---")
    ets = EffectTimeSeries().fit(control, treatment, all_ts)
    r = ets.result()
    for tp in r.time_points[:3]:
        print(f"  Day {int(tp['timestamp']) + 1}: effect={tp['cumulative_lift']:+.4f}, p={tp['pvalue']:.4f}")
    print(f"  ... ({len(r.time_points)} total time points)")
    print(f"  Final lift: {r.final_lift:+.4f}, stable: {r.is_stable}")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  splita + Kaggle Dataset Patterns")
    print("  Showing idiomatic pandas workflows")
    print("=" * 60)

    kaggle_ab_testing_pattern()
    kaggle_cookie_cats_pattern()
    ecommerce_revenue_pattern()
    timeseries_dashboard_pattern()

    print("\n" + "=" * 60)
    print("  ALL PATTERNS COMPLETED SUCCESSFULLY")
    print("=" * 60)
