"""Tests for splita.datasets — sample dataset generators."""

from __future__ import annotations

import numpy as np
import pytest

from splita.datasets import (
    load_ecommerce,
    load_marketplace,
    load_mobile_app,
    load_subscription,
)


# ─── E-commerce dataset ──────────────────────────────────────────────


class TestLoadEcommerce:
    """Tests for the e-commerce dataset generator."""

    def test_returns_dict_with_expected_keys(self):
        data = load_ecommerce()
        expected_keys = {
            "control", "treatment", "pre_control", "pre_treatment",
            "timestamps", "user_segments", "description",
        }
        assert set(data.keys()) == expected_keys

    def test_control_and_treatment_are_numpy_arrays(self):
        data = load_ecommerce()
        assert isinstance(data["control"], np.ndarray)
        assert isinstance(data["treatment"], np.ndarray)

    def test_has_5000_users_per_group(self):
        data = load_ecommerce()
        assert len(data["control"]) == 5000
        assert len(data["treatment"]) == 5000

    def test_revenue_is_non_negative(self):
        data = load_ecommerce()
        assert (data["control"] >= 0).all()
        assert (data["treatment"] >= 0).all()

    def test_timestamps_within_28_days(self):
        data = load_ecommerce()
        assert data["timestamps"].min() >= 0
        assert data["timestamps"].max() < 28

    def test_user_segments_are_valid(self):
        data = load_ecommerce()
        valid = {"new", "returning", "loyal"}
        assert set(data["user_segments"]).issubset(valid)

    def test_description_is_nonempty_string(self):
        data = load_ecommerce()
        assert isinstance(data["description"], str)
        assert len(data["description"]) > 20

    def test_deterministic_with_same_seed(self):
        d1 = load_ecommerce(seed=42)
        d2 = load_ecommerce(seed=42)
        np.testing.assert_array_equal(d1["control"], d2["control"])

    def test_different_seed_gives_different_data(self):
        d1 = load_ecommerce(seed=42)
        d2 = load_ecommerce(seed=99)
        assert not np.array_equal(d1["control"], d2["control"])

    def test_pre_experiment_covariates_are_non_negative(self):
        data = load_ecommerce()
        assert (data["pre_control"] >= 0).all()
        assert (data["pre_treatment"] >= 0).all()


# ─── Marketplace dataset ─────────────────────────────────────────────


class TestLoadMarketplace:
    """Tests for the marketplace dataset generator."""

    def test_returns_dict_with_expected_keys(self):
        data = load_marketplace()
        expected_keys = {
            "buyer_control", "buyer_treatment",
            "seller_control", "seller_treatment",
            "buyer_sessions", "seller_sessions",
            "description",
        }
        assert set(data.keys()) == expected_keys

    def test_buyer_arrays_have_correct_length(self):
        data = load_marketplace()
        assert len(data["buyer_control"]) == 3000
        assert len(data["buyer_treatment"]) == 3000

    def test_seller_arrays_have_correct_length(self):
        data = load_marketplace()
        assert len(data["seller_control"]) == 800
        assert len(data["seller_treatment"]) == 800

    def test_buyer_spend_is_non_negative(self):
        data = load_marketplace()
        assert (data["buyer_control"] >= 0).all()
        assert (data["buyer_treatment"] >= 0).all()

    def test_seller_listings_are_non_negative(self):
        data = load_marketplace()
        assert (data["seller_control"] >= 0).all()
        assert (data["seller_treatment"] >= 0).all()

    def test_deterministic_output(self):
        d1 = load_marketplace(seed=123)
        d2 = load_marketplace(seed=123)
        np.testing.assert_array_equal(d1["buyer_control"], d2["buyer_control"])

    def test_description_is_nonempty_string(self):
        data = load_marketplace()
        assert isinstance(data["description"], str)
        assert "marketplace" in data["description"].lower()


# ─── Subscription dataset ────────────────────────────────────────────


class TestLoadSubscription:
    """Tests for the subscription dataset generator."""

    def test_returns_dict_with_expected_keys(self):
        data = load_subscription()
        expected_keys = {
            "control", "treatment",
            "control_churned", "treatment_churned",
            "pre_control", "pre_treatment",
            "control_plan", "treatment_plan",
            "description",
        }
        assert set(data.keys()) == expected_keys

    def test_has_2000_users_per_group(self):
        data = load_subscription()
        assert len(data["control"]) == 2000
        assert len(data["treatment"]) == 2000

    def test_time_values_are_positive(self):
        data = load_subscription()
        assert (data["control"] > 0).all()
        assert (data["treatment"] > 0).all()

    def test_time_values_capped_at_90_days(self):
        data = load_subscription()
        assert data["control"].max() <= 90.0
        assert data["treatment"].max() <= 90.0

    def test_churn_flags_are_binary(self):
        data = load_subscription()
        assert set(np.unique(data["control_churned"])).issubset({0, 1})
        assert set(np.unique(data["treatment_churned"])).issubset({0, 1})

    def test_some_users_churned(self):
        data = load_subscription()
        assert data["control_churned"].sum() > 0
        assert data["treatment_churned"].sum() > 0

    def test_some_users_censored(self):
        """Not all users should have churned within 90 days."""
        data = load_subscription()
        assert (data["control_churned"] == 0).sum() > 0

    def test_plan_types_are_valid(self):
        data = load_subscription()
        valid = {"free", "basic", "pro"}
        assert set(data["control_plan"]).issubset(valid)
        assert set(data["treatment_plan"]).issubset(valid)

    def test_deterministic_output(self):
        d1 = load_subscription(seed=7)
        d2 = load_subscription(seed=7)
        np.testing.assert_array_equal(d1["control"], d2["control"])

    def test_feature_adoption_scores_in_range(self):
        data = load_subscription()
        assert data["pre_control"].min() >= 0
        assert data["pre_control"].max() <= 10


# ─── Mobile app dataset ──────────────────────────────────────────────


class TestLoadMobileApp:
    """Tests for the mobile app dataset generator."""

    def test_returns_dict_with_expected_keys(self):
        data = load_mobile_app()
        expected_keys = {
            "control", "treatment",
            "control_sessions", "treatment_sessions",
            "control_purchases", "treatment_purchases",
            "pre_control", "pre_treatment",
            "user_tenure_days",
            "description",
        }
        assert set(data.keys()) == expected_keys

    def test_has_4000_users(self):
        data = load_mobile_app()
        assert len(data["control"]) == 4000
        assert len(data["treatment"]) == 4000

    def test_session_minutes_are_non_negative(self):
        data = load_mobile_app()
        assert (data["control"] >= 0).all()
        assert (data["treatment"] >= 0).all()

    def test_session_counts_are_non_negative(self):
        data = load_mobile_app()
        assert (data["control_sessions"] >= 0).all()
        assert (data["treatment_sessions"] >= 0).all()

    def test_purchases_are_non_negative(self):
        data = load_mobile_app()
        assert (data["control_purchases"] >= 0).all()
        assert (data["treatment_purchases"] >= 0).all()

    def test_tenure_within_range(self):
        data = load_mobile_app()
        assert data["user_tenure_days"].min() >= 1
        assert data["user_tenure_days"].max() <= 365

    def test_deterministic_output(self):
        d1 = load_mobile_app(seed=99)
        d2 = load_mobile_app(seed=99)
        np.testing.assert_array_equal(d1["control"], d2["control"])

    def test_description_mentions_mobile(self):
        data = load_mobile_app()
        assert "mobile" in data["description"].lower()

    def test_pre_experiment_data_matches_group_size(self):
        data = load_mobile_app()
        assert len(data["pre_control"]) == 4000
        assert len(data["pre_treatment"]) == 4000
