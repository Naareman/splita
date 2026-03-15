"""Deterministic sample dataset generators for splita tutorials.

All generators use seeded RNGs so they produce identical data on every
call, making them suitable for reproducible examples and documentation.
"""

from __future__ import annotations

import numpy as np


def load_ecommerce(*, seed: int = 42) -> dict:
    """Generate a realistic e-commerce A/B test dataset.

    Simulates an online store testing a new checkout flow.  Revenue
    follows a heavy-tailed log-normal distribution with realistic
    day-of-week effects (weekend lift).  Conversion rate is ~8% control
    vs ~9.5% treatment.

    Parameters
    ----------
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        - ``control`` : np.ndarray — revenue per user (0 if no purchase)
        - ``treatment`` : np.ndarray — revenue per user (0 if no purchase)
        - ``pre_control`` : np.ndarray — pre-experiment page views
        - ``pre_treatment`` : np.ndarray — pre-experiment page views
        - ``timestamps`` : np.ndarray — integer day index (0-27)
        - ``user_segments`` : np.ndarray — segment labels ('new', 'returning', 'loyal')
        - ``description`` : str — human-readable scenario description

    Examples
    --------
    >>> data = load_ecommerce()
    >>> len(data["control"])
    5000
    >>> data["control"].dtype
    dtype('float64')
    """
    rng = np.random.default_rng(seed)
    n = 5000

    # Segments
    segments = rng.choice(["new", "returning", "loyal"], size=n, p=[0.5, 0.35, 0.15])

    # Day-of-week: 28-day experiment
    days = rng.integers(0, 28, size=n)
    day_of_week = days % 7  # 0=Mon, 6=Sun

    # Weekend lift on conversion
    weekend_boost = np.where((day_of_week >= 5), 0.015, 0.0)

    # Segment-based conversion rates
    segment_rates = {"new": 0.06, "returning": 0.09, "loyal": 0.14}
    base_rates = np.array([segment_rates[s] for s in segments])

    # Control: base conversion
    ctrl_prob = base_rates + weekend_boost
    ctrl_convert = rng.binomial(1, np.clip(ctrl_prob, 0, 1), size=n)

    # Treatment: +1.5pp uplift
    trt_prob = base_rates + 0.015 + weekend_boost
    trt_convert = rng.binomial(1, np.clip(trt_prob, 0, 1), size=n)

    # Revenue: log-normal for converters (heavy-tailed)
    ctrl_revenue = np.where(
        ctrl_convert == 1,
        np.clip(rng.lognormal(mean=3.5, sigma=0.8, size=n), 5, 500),
        0.0,
    )
    trt_revenue = np.where(
        trt_convert == 1,
        np.clip(rng.lognormal(mean=3.5, sigma=0.8, size=n), 5, 500),
        0.0,
    )

    # Pre-experiment covariate: page views (correlated with conversion)
    pre_ctrl = rng.poisson(5, size=n).astype(float) + np.where(
        ctrl_convert == 1, rng.poisson(3, size=n), 0
    ).astype(float)
    pre_trt = rng.poisson(5, size=n).astype(float) + np.where(
        trt_convert == 1, rng.poisson(3, size=n), 0
    ).astype(float)

    return {
        "control": ctrl_revenue,
        "treatment": trt_revenue,
        "pre_control": pre_ctrl,
        "pre_treatment": pre_trt,
        "timestamps": days,
        "user_segments": segments,
        "description": (
            "E-commerce A/B test: new checkout flow vs. existing.\n"
            "5,000 users per group over 28 days.\n"
            "Revenue is heavy-tailed (log-normal), with weekend lift.\n"
            "Segments: new (50%), returning (35%), loyal (15%).\n"
            "Expected effect: +1.5pp conversion uplift (~19% relative)."
        ),
    }


def load_marketplace(*, seed: int = 123) -> dict:
    """Generate a two-sided marketplace dataset with buyer/seller data.

    Simulates a marketplace testing a new search ranking algorithm.
    Both buyer conversion and seller listing completions are tracked.
    Buyer spend follows a Pareto-like distribution.

    Parameters
    ----------
    seed : int, default 123
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        - ``buyer_control`` : np.ndarray — buyer spend (0 if no purchase)
        - ``buyer_treatment`` : np.ndarray — buyer spend
        - ``seller_control`` : np.ndarray — seller listings completed
        - ``seller_treatment`` : np.ndarray — seller listings completed
        - ``buyer_sessions`` : np.ndarray — number of sessions per buyer
        - ``seller_sessions`` : np.ndarray — number of sessions per seller
        - ``description`` : str — scenario description

    Examples
    --------
    >>> data = load_marketplace()
    >>> len(data["buyer_control"])
    3000
    """
    rng = np.random.default_rng(seed)
    n_buyers = 3000
    n_sellers = 800

    # Buyers
    buyer_sessions_ctrl = rng.poisson(3, n_buyers).astype(float) + 1
    buyer_sessions_trt = rng.poisson(3, n_buyers).astype(float) + 1

    # Buyer conversion (higher with more sessions)
    buyer_conv_ctrl = rng.binomial(
        1, np.clip(0.12 + 0.02 * np.log(buyer_sessions_ctrl), 0, 0.5), n_buyers
    )
    buyer_conv_trt = rng.binomial(
        1, np.clip(0.14 + 0.02 * np.log(buyer_sessions_trt), 0, 0.5), n_buyers
    )

    # Buyer spend: Pareto-like (power law)
    buyer_spend_ctrl = np.where(
        buyer_conv_ctrl == 1,
        np.clip((rng.pareto(2.5, n_buyers) + 1) * 20, 10, 2000),
        0.0,
    )
    buyer_spend_trt = np.where(
        buyer_conv_trt == 1,
        np.clip((rng.pareto(2.5, n_buyers) + 1) * 20, 10, 2000),
        0.0,
    )

    # Sellers: listing completions (count data)
    seller_sessions_ctrl = rng.poisson(5, n_sellers).astype(float) + 1
    rng.poisson(5, n_sellers).astype(float) + 1

    seller_listings_ctrl = rng.poisson(2.0, n_sellers).astype(float)
    seller_listings_trt = rng.poisson(2.3, n_sellers).astype(float)  # 15% lift

    return {
        "buyer_control": buyer_spend_ctrl,
        "buyer_treatment": buyer_spend_trt,
        "seller_control": seller_listings_ctrl,
        "seller_treatment": seller_listings_trt,
        "buyer_sessions": buyer_sessions_ctrl,
        "seller_sessions": seller_sessions_ctrl,
        "description": (
            "Two-sided marketplace A/B test: new search ranking algorithm.\n"
            "3,000 buyers and 800 sellers per group.\n"
            "Buyer spend follows a Pareto distribution (heavy right tail).\n"
            "Seller listings are count data (Poisson).\n"
            "Expected effect: +2pp buyer conversion, +15% seller listings."
        ),
    }


def load_subscription(*, seed: int = 7) -> dict:
    """Generate a subscription product dataset with churn events.

    Simulates a SaaS product testing a new onboarding flow.  Tracks
    time-to-churn (survival data) and feature adoption as a covariate.

    Parameters
    ----------
    seed : int, default 7
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        - ``control`` : np.ndarray — days active before churn (or censored)
        - ``treatment`` : np.ndarray — days active before churn (or censored)
        - ``control_churned`` : np.ndarray — 1 if churned, 0 if censored (still active)
        - ``treatment_churned`` : np.ndarray — 1 if churned, 0 if censored
        - ``pre_control`` : np.ndarray — feature adoption score (0-10)
        - ``pre_treatment`` : np.ndarray — feature adoption score (0-10)
        - ``control_plan`` : np.ndarray — plan type ('free', 'basic', 'pro')
        - ``treatment_plan`` : np.ndarray — plan type
        - ``description`` : str — scenario description

    Examples
    --------
    >>> data = load_subscription()
    >>> len(data["control"])
    2000
    >>> data["control_churned"].sum() > 0
    True
    """
    rng = np.random.default_rng(seed)
    n = 2000

    # Plan types
    plans_ctrl = rng.choice(["free", "basic", "pro"], size=n, p=[0.6, 0.3, 0.1])
    plans_trt = rng.choice(["free", "basic", "pro"], size=n, p=[0.6, 0.3, 0.1])

    # Plan-based churn hazard
    plan_hazard = {"free": 0.03, "basic": 0.015, "pro": 0.008}

    # Feature adoption (pre-experiment covariate)
    pre_ctrl = np.clip(rng.normal(4.0, 2.0, n), 0, 10)
    pre_trt = np.clip(rng.normal(4.0, 2.0, n), 0, 10)

    # Time-to-churn: exponential with plan-dependent rate
    # Treatment reduces hazard by ~15%
    hazard_ctrl = np.array([plan_hazard[p] for p in plans_ctrl])
    hazard_trt = np.array([plan_hazard[p] for p in plans_trt]) * 0.85

    # Adoption score reduces hazard
    hazard_ctrl = hazard_ctrl * np.exp(-0.05 * pre_ctrl)
    hazard_trt = hazard_trt * np.exp(-0.05 * pre_trt)

    time_ctrl = rng.exponential(1.0 / np.clip(hazard_ctrl, 1e-6, None), n)
    time_trt = rng.exponential(1.0 / np.clip(hazard_trt, 1e-6, None), n)

    # Censoring at 90 days (observation window)
    censor_time = 90.0
    churned_ctrl = (time_ctrl <= censor_time).astype(int)
    churned_trt = (time_trt <= censor_time).astype(int)
    time_ctrl = np.minimum(time_ctrl, censor_time)
    time_trt = np.minimum(time_trt, censor_time)

    return {
        "control": time_ctrl,
        "treatment": time_trt,
        "control_churned": churned_ctrl,
        "treatment_churned": churned_trt,
        "pre_control": pre_ctrl,
        "pre_treatment": pre_trt,
        "control_plan": plans_ctrl,
        "treatment_plan": plans_trt,
        "description": (
            "SaaS subscription A/B test: new onboarding flow.\n"
            "2,000 users per group observed for 90 days.\n"
            "Time-to-churn follows an exponential distribution.\n"
            "Plans: free (60%), basic (30%), pro (10%).\n"
            "Expected effect: 15% reduction in churn hazard.\n"
            "Right-censored at 90 days (users still active)."
        ),
    }


def load_mobile_app(*, seed: int = 99) -> dict:
    """Generate a mobile app engagement dataset with session-level data.

    Simulates a mobile app testing a new recommendation engine.
    Tracks session duration, sessions per day, and in-app purchases.
    Includes day-of-week effects and user tenure as covariates.

    Parameters
    ----------
    seed : int, default 99
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        - ``control`` : np.ndarray — total session minutes per user (14-day window)
        - ``treatment`` : np.ndarray — total session minutes per user
        - ``control_sessions`` : np.ndarray — number of sessions per user
        - ``treatment_sessions`` : np.ndarray — number of sessions per user
        - ``control_purchases`` : np.ndarray — in-app purchase count per user
        - ``treatment_purchases`` : np.ndarray — in-app purchase count per user
        - ``pre_control`` : np.ndarray — sessions in 7 days before experiment
        - ``pre_treatment`` : np.ndarray — sessions in 7 days before experiment
        - ``user_tenure_days`` : np.ndarray — days since install
        - ``description`` : str — scenario description

    Examples
    --------
    >>> data = load_mobile_app()
    >>> len(data["control"])
    4000
    >>> (data["control"] >= 0).all()
    True
    """
    rng = np.random.default_rng(seed)
    n = 4000

    # User tenure (days since install) - exponential decay (most users are new)
    tenure = np.clip(rng.exponential(60, n), 1, 365).astype(int)

    # Tenure affects baseline engagement
    tenure_factor = np.log1p(tenure) / np.log1p(365)  # 0 to 1

    # Sessions per 14-day window (negative binomial for overdispersion)
    base_sessions = 5 + 10 * tenure_factor
    ctrl_sessions = rng.negative_binomial(3, 3 / (3 + base_sessions), n).astype(float)
    trt_sessions = rng.negative_binomial(
        3,
        3 / (3 + base_sessions * 1.08),
        n,  # 8% session lift
    ).astype(float)

    # Session duration: gamma distribution (right-skewed)
    # Average minutes per session
    avg_duration = 4.0 + 2.0 * tenure_factor
    ctrl_total_mins = np.zeros(n)
    trt_total_mins = np.zeros(n)
    for i in range(n):
        if ctrl_sessions[i] > 0:
            ctrl_total_mins[i] = rng.gamma(2.0, avg_duration[i] / 2.0, int(ctrl_sessions[i])).sum()
        if trt_sessions[i] > 0:
            trt_total_mins[i] = rng.gamma(
                2.0, avg_duration[i] * 1.05 / 2.0, int(trt_sessions[i])
            ).sum()

    # In-app purchases: rare event (Poisson with low rate)
    purchase_rate_ctrl = 0.3 * (1 + 0.5 * tenure_factor)
    purchase_rate_trt = 0.35 * (1 + 0.5 * tenure_factor)  # ~17% purchase lift
    ctrl_purchases = rng.poisson(purchase_rate_ctrl, n).astype(float)
    trt_purchases = rng.poisson(purchase_rate_trt, n).astype(float)

    # Pre-experiment sessions (7-day window)
    pre_ctrl = rng.negative_binomial(2, 2 / (2 + base_sessions * 0.5), n).astype(float)
    pre_trt = rng.negative_binomial(2, 2 / (2 + base_sessions * 0.5), n).astype(float)

    return {
        "control": ctrl_total_mins,
        "treatment": trt_total_mins,
        "control_sessions": ctrl_sessions,
        "treatment_sessions": trt_sessions,
        "control_purchases": ctrl_purchases,
        "treatment_purchases": trt_purchases,
        "pre_control": pre_ctrl,
        "pre_treatment": pre_trt,
        "user_tenure_days": tenure,
        "description": (
            "Mobile app A/B test: new recommendation engine.\n"
            "4,000 users per group over a 14-day window.\n"
            "Session count uses negative binomial (overdispersed).\n"
            "Session duration uses gamma distribution (right-skewed).\n"
            "Tenure affects baseline engagement (log-scaled).\n"
            "Expected effects: +8% sessions, +5% duration, +17% purchases."
        ),
    }
