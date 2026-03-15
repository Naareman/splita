"""Streamlit app for the splita interactive playground.

This file is launched by :func:`splita.playground` and should not be
imported directly.
"""

from __future__ import annotations

import contextlib

import numpy as np
import streamlit as st

import splita
from splita.datasets import load_ecommerce, load_marketplace, load_mobile_app, load_subscription

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="splita playground",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("splita playground")
mode = st.sidebar.radio(
    "Analysis mode",
    ["A/B Test", "Bayesian", "Power Planning", "Sequential", "Sample Data"],
)

# ── Helpers ──────────────────────────────────────────────────────────

_DATASET_LOADERS = {
    "E-commerce": load_ecommerce,
    "Marketplace": load_marketplace,
    "Subscription": load_subscription,
    "Mobile App": load_mobile_app,
}


def _parse_csv_values(text: str) -> np.ndarray | None:
    """Parse a comma-separated string into a NumPy array."""
    text = text.strip()
    if not text:
        return None
    try:
        return np.array([float(x.strip()) for x in text.split(",") if x.strip()])
    except ValueError:
        st.error("Could not parse input. Ensure values are comma-separated numbers.")
        return None


def _load_dataset(name: str) -> dict:
    """Load a sample dataset by name."""
    return _DATASET_LOADERS[name]()


def _get_data_from_csv(
    uploaded_file, control_col: str, treatment_col: str, pre_col: str | None = None
):
    """Extract arrays from uploaded CSV columns."""
    import pandas as pd

    df = pd.read_csv(uploaded_file)
    ctrl = df[control_col].dropna().values.astype(float)
    trt = df[treatment_col].dropna().values.astype(float)
    pre_ctrl = None
    pre_trt = None
    if pre_col and pre_col != "None":
        pre = df[pre_col].dropna().values.astype(float)
        half = len(pre) // 2
        pre_ctrl = pre[:half]
        pre_trt = pre[half:]
    return ctrl, trt, pre_ctrl, pre_trt


def _display_check_result(check_result):
    """Render CheckResult as metrics."""
    col1, col2, col3 = st.columns(3)
    col1.metric("SRM", "PASS" if check_result.srm_passed else "FAIL")
    col2.metric("Outliers Detected", "Yes" if check_result.has_outliers else "No")
    col3.metric("Adequately Powered", "Yes" if check_result.is_powered else "No")

    if check_result.recommendations:
        with st.expander("Recommendations"):
            for rec in check_result.recommendations:
                st.markdown(f"- {rec}")


def _display_experiment_result(result):
    """Render an ExperimentResult with metrics and explanation."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lift", f"{result.lift:+.4f}")
    col2.metric("Relative Lift", f"{result.relative_lift:+.2%}")
    col3.metric("p-value", f"{result.pvalue:.4f}")
    col4.metric("Significant", "Yes" if result.significant else "No")

    st.markdown(
        f"**Method:** {result.method} | "
        f"**CI:** [{result.ci_lower:.4f}, {result.ci_upper:.4f}] | "
        f"**Alpha:** {result.alpha}"
    )


def _display_forest_plot(results):
    """Display a forest plot if matplotlib is available."""
    try:
        fig = splita.viz.forest_plot(results)
        st.pyplot(fig)
    except ImportError:
        st.warning("Install matplotlib for visualizations: pip install splita[viz]")
    except Exception as exc:
        st.warning(f"Could not generate plot: {exc}")


def _display_export(result):
    """Offer download buttons for HTML report and JSON."""
    col1, col2 = st.columns(2)
    try:
        html_report = splita.report(result, title="splita playground report")
        col1.download_button(
            "Download HTML Report",
            html_report,
            file_name="report.html",
            mime="text/html",
        )
    except Exception:
        pass

    try:
        json_str = result.to_json()
        col2.download_button(
            "Download JSON",
            json_str,
            file_name="result.json",
            mime="application/json",
        )
    except Exception:
        pass


# ── A/B Test Mode ────────────────────────────────────────────────────


def _ab_test_mode():
    st.title("A/B Test Analysis")
    st.markdown("Interactive frequentist A/B test analysis — no code required.")

    input_method = st.radio("Data input", ["Upload CSV", "Use sample data", "Enter manually"])

    ctrl = trt = pre_ctrl = pre_trt = None

    if input_method == "Upload CSV":
        import pandas as pd

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            cols = list(df.columns)
            control_col = st.selectbox("Control column", cols, index=0)
            treatment_col = st.selectbox("Treatment column", cols, index=min(1, len(cols) - 1))
            pre_options = ["None", *cols]
            pre_col = st.selectbox("Pre-experiment column (optional)", pre_options, index=0)
            ctrl = df[control_col].dropna().values.astype(float)
            trt = df[treatment_col].dropna().values.astype(float)
            if pre_col != "None":
                pre_vals = df[pre_col].dropna().values.astype(float)
                # Split pre-data to match control/treatment lengths
                pre_ctrl = pre_vals[: len(ctrl)]
                pre_trt = pre_vals[: len(trt)]

    elif input_method == "Use sample data":
        dataset_name = st.selectbox("Dataset", list(_DATASET_LOADERS.keys()))
        data = _load_dataset(dataset_name)
        st.info(data["description"])
        ctrl = data["control"]
        trt = data["treatment"]
        if "pre_control" in data:
            pre_ctrl = data["pre_control"]
        if "pre_treatment" in data:
            pre_trt = data["pre_treatment"]

    elif input_method == "Enter manually":
        control_text = st.text_area(
            "Control data (comma-separated)",
            placeholder="e.g. 0,1,0,0,1,1,0,0,0,1",
        )
        treatment_text = st.text_area(
            "Treatment data (comma-separated)",
            placeholder="e.g. 1,0,1,1,0,1,1,0,1,1",
        )
        ctrl = _parse_csv_values(control_text) if control_text else None
        trt = _parse_csv_values(treatment_text) if treatment_text else None

    # Advanced options
    with st.expander("Advanced options"):
        alpha = st.slider("Significance level (alpha)", 0.01, 0.20, 0.05, step=0.01)
        use_cuped = st.checkbox("Apply CUPED variance reduction", value=False)
        _use_outlier = st.checkbox("Handle outliers", value=True)
        _method = st.selectbox(
            "Test method",
            ["auto", "ttest", "ztest", "mannwhitney", "bootstrap"],
        )

    if st.button("Run Analysis", type="primary"):
        if ctrl is None or trt is None:
            st.error("Please provide both control and treatment data.")
            return

        if len(ctrl) < 2 or len(trt) < 2:
            st.error("Each group must have at least 2 observations.")
            return

        # 1. Data quality
        st.subheader("Data Quality")
        try:
            check_result = splita.check(ctrl, trt)
            _display_check_result(check_result)
        except Exception as exc:
            st.warning(f"Data quality check failed: {exc}")

        # 2. Run analysis
        st.subheader("Results")
        try:
            cuped_kwargs = {}
            if use_cuped and pre_ctrl is not None and pre_trt is not None:
                cuped_kwargs["control_pre"] = pre_ctrl
                cuped_kwargs["treatment_pre"] = pre_trt

            auto_result = splita.auto(ctrl, trt, alpha=alpha, **cuped_kwargs)
            primary = auto_result.primary_result
            _display_experiment_result(primary)

            # Pipeline steps
            if auto_result.pipeline_steps:
                with st.expander("Pipeline steps"):
                    for step in auto_result.pipeline_steps:
                        st.markdown(f"- {step}")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

        # 3. Plain English explanation
        st.subheader("Plain English")
        try:
            explanation = splita.explain(primary)
            st.info(explanation)
        except Exception as exc:
            st.warning(f"Could not generate explanation: {exc}")

        # 4. Visualization
        st.subheader("Visualization")
        _display_forest_plot([primary])

        # 5. Export
        st.subheader("Export")
        _display_export(primary)


# ── Bayesian Mode ────────────────────────────────────────────────────


def _bayesian_mode():
    st.title("Bayesian A/B Test")
    st.markdown("Bayesian analysis with posterior probabilities and credible intervals.")

    input_method = st.radio("Data input", ["Use sample data", "Enter manually"], key="bayes_input")

    ctrl = trt = None

    if input_method == "Use sample data":
        dataset_name = st.selectbox("Dataset", list(_DATASET_LOADERS.keys()), key="bayes_ds")
        data = _load_dataset(dataset_name)
        st.info(data["description"])
        ctrl = data["control"]
        trt = data["treatment"]

    elif input_method == "Enter manually":
        control_text = st.text_area(
            "Control data (comma-separated)",
            placeholder="e.g. 0,1,0,0,1,1,0,0,0,1",
            key="bayes_ctrl",
        )
        treatment_text = st.text_area(
            "Treatment data (comma-separated)",
            placeholder="e.g. 1,0,1,1,0,1,1,0,1,1",
            key="bayes_trt",
        )
        ctrl = _parse_csv_values(control_text) if control_text else None
        trt = _parse_csv_values(treatment_text) if treatment_text else None

    # ROPE option
    with st.expander("Advanced options"):
        use_rope = st.checkbox("Use ROPE (Region of Practical Equivalence)")
        rope_lower = st.number_input("ROPE lower", value=-0.01, format="%.4f")
        rope_upper = st.number_input("ROPE upper", value=0.01, format="%.4f")
        n_samples = st.number_input("Posterior samples", value=50000, min_value=1000, step=10000)

    if st.button("Run Bayesian Analysis", type="primary"):
        if ctrl is None or trt is None:
            st.error("Please provide both control and treatment data.")
            return

        if len(ctrl) < 2 or len(trt) < 2:
            st.error("Each group must have at least 2 observations.")
            return

        rope = (rope_lower, rope_upper) if use_rope else None

        try:
            bayes = splita.BayesianExperiment(
                ctrl, trt, rope=rope, n_samples=int(n_samples), random_state=42
            )
            result = bayes.run()

            st.subheader("Posterior Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("P(B > A)", f"{result.prob_b_beats_a:.4f}")
            col2.metric("Expected Loss (A)", f"{result.expected_loss_a:.6f}")
            col3.metric("Expected Loss (B)", f"{result.expected_loss_b:.6f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Lift", f"{result.lift:+.4f}")
            col5.metric("Relative Lift", f"{result.relative_lift:+.2%}")
            col6.metric(
                "95% Credible Interval",
                f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
            )

            if hasattr(result, "prob_in_rope") and result.prob_in_rope is not None:
                st.metric("P(effect in ROPE)", f"{result.prob_in_rope:.4f}")

            # Explanation
            st.subheader("Plain English")
            try:
                explanation = splita.explain(result)
                st.info(explanation)
            except Exception:
                pass

            # Export
            st.subheader("Export")
            _display_export(result)

        except Exception as exc:
            st.error(f"Bayesian analysis failed: {exc}")


# ── Power Planning Mode ──────────────────────────────────────────────


def _power_planning_mode():
    st.title("Power Planning")
    st.markdown("Plan your experiment sample size with real-time power calculations.")

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.slider(
            "Baseline rate",
            min_value=0.01,
            max_value=0.99,
            value=0.10,
            step=0.01,
            help="Current conversion/metric rate (e.g. 0.10 for 10%)",
        )
        mde_pct = st.slider(
            "Minimum Detectable Effect (relative %)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Smallest relative change you want to detect",
        )
    with col2:
        power = st.slider(
            "Statistical power",
            min_value=0.50,
            max_value=0.99,
            value=0.80,
            step=0.01,
        )
        alpha = st.slider(
            "Significance level (alpha)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="power_alpha",
        )

    daily_users = st.number_input(
        "Daily users (total across variants)",
        min_value=10,
        value=10000,
        step=1000,
    )

    metric = st.selectbox("Metric type", ["conversion", "continuous"])

    mde_abs = baseline * (mde_pct / 100.0)

    # Always show results
    st.subheader("Sample Size Estimate")

    try:
        ss = splita.SampleSize(
            baseline=baseline,
            mde=mde_abs,
            alpha=alpha,
            power=power,
            metric=metric,
        )
        result = ss.run()
        col1, col2, col3 = st.columns(3)
        col1.metric("Per-variant sample size", f"{result.n_per_variant:,}")
        col2.metric("Total sample size", f"{result.n_total:,}")
        days_needed = max(1, result.n_total / daily_users)
        col3.metric("Estimated days", f"{days_needed:.0f}")
    except Exception as exc:
        st.error(f"Sample size calculation failed: {exc}")

    # Power report
    st.subheader("Power Report")
    try:
        txt = splita.power_report(baseline, metric=metric, alpha=alpha, format="text")
        st.code(txt, language="text")
    except Exception as exc:
        st.warning(f"Could not generate power report: {exc}")

    # Power curve
    st.subheader("Power Curve")
    try:
        fig = splita.viz.power_curve(
            baseline=baseline,
            mde=mde_abs,
            alpha=alpha,
            metric=metric,
        )
        st.pyplot(fig)
    except ImportError:
        st.warning("Install matplotlib for the power curve: pip install splita[viz]")
    except Exception as exc:
        st.warning(f"Could not generate power curve: {exc}")


# ── Sequential Mode ──────────────────────────────────────────────────


def _sequential_mode():
    st.title("Sequential Testing (mSPRT)")
    st.markdown("Monitor your experiment with always-valid p-values. No peeking penalty.")

    input_method = st.radio(
        "Data input",
        ["Use sample data", "Enter manually"],
        key="seq_input",
    )

    ctrl = trt = None

    if input_method == "Use sample data":
        dataset_name = st.selectbox("Dataset", list(_DATASET_LOADERS.keys()), key="seq_ds")
        data = _load_dataset(dataset_name)
        st.info(data["description"])
        ctrl = data["control"]
        trt = data["treatment"]

    elif input_method == "Enter manually":
        control_text = st.text_area(
            "Control data (comma-separated)",
            placeholder="e.g. 0,1,0,0,1,1,0,0,0,1",
            key="seq_ctrl",
        )
        treatment_text = st.text_area(
            "Treatment data (comma-separated)",
            placeholder="e.g. 1,0,1,1,0,1,1,0,1,1",
            key="seq_trt",
        )
        ctrl = _parse_csv_values(control_text) if control_text else None
        trt = _parse_csv_values(treatment_text) if treatment_text else None

    with st.expander("Advanced options"):
        metric_type = st.selectbox("Metric type", ["conversion", "continuous"], key="seq_metric")
        seq_alpha = st.slider("Significance level", 0.01, 0.20, 0.05, step=0.01, key="seq_alpha")
        n_batches = st.slider(
            "Number of batches (simulated peeks)",
            min_value=2,
            max_value=50,
            value=10,
            help="Split data into this many batches to simulate sequential monitoring",
        )

    if st.button("Run Sequential Analysis", type="primary"):
        if ctrl is None or trt is None:
            st.error("Please provide both control and treatment data.")
            return

        if len(ctrl) < 2 or len(trt) < 2:
            st.error("Each group must have at least 2 observations.")
            return

        try:
            test = splita.mSPRT(metric=metric_type, alpha=seq_alpha)

            # Split into batches
            batch_size_ctrl = max(1, len(ctrl) // n_batches)
            batch_size_trt = max(1, len(trt) // n_batches)

            states = []
            for i in range(n_batches):
                start_c = i * batch_size_ctrl
                end_c = min((i + 1) * batch_size_ctrl, len(ctrl))
                start_t = i * batch_size_trt
                end_t = min((i + 1) * batch_size_trt, len(trt))

                if start_c >= len(ctrl) or start_t >= len(trt):
                    break

                state = test.update(ctrl[start_c:end_c], trt[start_t:end_t])
                states.append(state)

            if not states:
                st.error("No batches could be formed from the data.")
                return

            # Final result
            final = test.result()

            st.subheader("Final Result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Always-Valid p-value", f"{final.always_valid_pvalue:.4f}")
            col2.metric("Effect Estimate", f"{final.current_effect_estimate:+.4f}")
            col3.metric("Should Stop", "Yes" if final.should_stop else "No")

            col4, col5 = st.columns(2)
            col4.metric(
                "Always-Valid CI",
                f"[{final.always_valid_ci_lower:.4f}, {final.always_valid_ci_upper:.4f}]",
            )
            col5.metric("Stopping Reason", final.stopping_reason)

            # Effect over time chart
            st.subheader("Sequential Monitoring")
            if states:
                import pandas as pd

                monitor_data = pd.DataFrame(
                    {
                        "Batch": list(range(1, len(states) + 1)),
                        "Always-Valid p-value": [s.always_valid_pvalue for s in states],
                        "Effect Estimate": [s.current_effect_estimate for s in states],
                        "CI Lower": [s.always_valid_ci_lower for s in states],
                        "CI Upper": [s.always_valid_ci_upper for s in states],
                        "N (total)": [s.n_control + s.n_treatment for s in states],
                    }
                )
                st.dataframe(monitor_data)

                st.line_chart(
                    monitor_data.set_index("Batch")[["Always-Valid p-value"]],
                )
                st.line_chart(
                    monitor_data.set_index("Batch")[["Effect Estimate", "CI Lower", "CI Upper"]],
                )

            # Export
            st.subheader("Export")
            _display_export(final)

        except Exception as exc:
            st.error(f"Sequential analysis failed: {exc}")


# ── Sample Data Mode ─────────────────────────────────────────────────


def _sample_data_mode():
    st.title("Sample Data Explorer")
    st.markdown("Load any built-in dataset and run a full analysis.")

    dataset_name = st.selectbox("Dataset", list(_DATASET_LOADERS.keys()), key="sample_ds")
    data = _load_dataset(dataset_name)

    st.subheader("Dataset Description")
    st.info(data["description"])

    st.subheader("Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Control**")
        st.write(f"n = {len(data['control']):,}")
        st.write(f"Mean = {np.mean(data['control']):.4f}")
        st.write(f"Std = {np.std(data['control']):.4f}")
    with col2:
        st.markdown("**Treatment**")
        st.write(f"n = {len(data['treatment']):,}")
        st.write(f"Mean = {np.mean(data['treatment']):.4f}")
        st.write(f"Std = {np.std(data['treatment']):.4f}")

    if st.button("Run Full Analysis", type="primary"):
        ctrl = data["control"]
        trt = data["treatment"]
        pre_ctrl = data.get("pre_control")
        pre_trt = data.get("pre_treatment")

        # Data quality
        st.subheader("Data Quality")
        try:
            check_result = splita.check(ctrl, trt)
            _display_check_result(check_result)
        except Exception as exc:
            st.warning(f"Data quality check failed: {exc}")

        # Auto analysis
        st.subheader("Frequentist Result")
        try:
            kwargs = {}
            if pre_ctrl is not None and pre_trt is not None:
                kwargs["control_pre"] = pre_ctrl
                kwargs["treatment_pre"] = pre_trt

            auto_result = splita.auto(ctrl, trt, **kwargs)
            primary = auto_result.primary_result
            _display_experiment_result(primary)

            if auto_result.pipeline_steps:
                with st.expander("Pipeline steps"):
                    for step in auto_result.pipeline_steps:
                        st.markdown(f"- {step}")
        except Exception as exc:
            st.error(f"Frequentist analysis failed: {exc}")
            primary = None

        # Bayesian analysis
        st.subheader("Bayesian Result")
        try:
            bayes = splita.BayesianExperiment(ctrl, trt, random_state=42)
            bayes_result = bayes.run()
            col1, col2, col3 = st.columns(3)
            col1.metric("P(B > A)", f"{bayes_result.prob_b_beats_a:.4f}")
            col2.metric("Expected Loss (B)", f"{bayes_result.expected_loss_b:.6f}")
            col3.metric(
                "95% Credible Interval",
                f"[{bayes_result.ci_lower:.4f}, {bayes_result.ci_upper:.4f}]",
            )
        except Exception as exc:
            st.warning(f"Bayesian analysis failed: {exc}")

        # Explanation
        if primary is not None:
            st.subheader("Plain English")
            with contextlib.suppress(Exception):
                st.info(splita.explain(primary))

        # Visualization
        if primary is not None:
            st.subheader("Visualization")
            _display_forest_plot([primary])

        # Export
        if primary is not None:
            st.subheader("Export")
            _display_export(primary)


# ── Mode router ──────────────────────────────────────────────────────

_MODES = {
    "A/B Test": _ab_test_mode,
    "Bayesian": _bayesian_mode,
    "Power Planning": _power_planning_mode,
    "Sequential": _sequential_mode,
    "Sample Data": _sample_data_mode,
}

_MODES[mode]()
