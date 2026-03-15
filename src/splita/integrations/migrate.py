"""Migrate experiment results from other platforms into splita objects.

Supports converting results from GrowthBook, Statsig, and generic
dictionaries into :class:`~splita._types.ExperimentResult`::

    from splita.integrations.migrate import migrate_from

    # From GrowthBook
    result = migrate_from(growthbook_data, platform="growthbook")

    # From Statsig
    result = migrate_from(statsig_data, platform="statsig")

    # From any platform with effect/pvalue/ci
    result = migrate_from(generic_data, platform="generic")
"""

from __future__ import annotations

from splita._types import ExperimentResult


def migrate_from(data: dict, platform: str = "growthbook") -> ExperimentResult:
    """Convert results from other platforms into a splita ExperimentResult.

    Parameters
    ----------
    data : dict
        The result data from the source platform.
    platform : {"growthbook", "statsig", "generic"}
        Which platform the data comes from.

    Returns
    -------
    ExperimentResult
        A splita result object populated from the input data.

    Raises
    ------
    ValueError
        If the platform is not supported or required keys are missing.
    """
    platform = platform.lower()
    if platform == "growthbook":
        return _from_growthbook(data)
    elif platform == "statsig":
        return _from_statsig(data)
    elif platform == "generic":
        return _from_generic(data)
    else:
        raise ValueError(
            f"Unsupported platform {platform!r}.\n"
            f"  Detail: supported platforms are 'growthbook', 'statsig', 'generic'.\n"
            f"  Hint: use platform='generic' for any dict with effect/pvalue/ci keys."
        )


def _require_keys(data: dict, keys: list[str], platform: str) -> None:
    """Check that all required keys are present."""
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(
            f"Missing required keys for {platform!r} migration: {missing}.\n"
            f"  Detail: got keys {list(data.keys())}.\n"
            f"  Hint: check the {platform} export format."
        )


def _from_growthbook(data: dict) -> ExperimentResult:
    """Convert GrowthBook Bayesian output to ExperimentResult.

    Expected keys: chance_to_win, effect, ci_lower, ci_upper,
    control_mean, treatment_mean, control_n, treatment_n.
    """
    _require_keys(
        data,
        ["chance_to_win", "effect", "ci_lower", "ci_upper"],
        "growthbook",
    )

    chance = float(data["chance_to_win"])
    effect = float(data["effect"])
    ci_lower = float(data["ci_lower"])
    ci_upper = float(data["ci_upper"])
    control_mean = float(data.get("control_mean", 0.0))
    treatment_mean = float(data.get("treatment_mean", control_mean + effect))
    control_n = int(data.get("control_n", 0))
    treatment_n = int(data.get("treatment_n", 0))

    # GrowthBook uses Bayesian chance_to_win; approximate a p-value
    pvalue = 2.0 * min(chance, 1.0 - chance)
    pvalue = max(0.0, min(1.0, pvalue))

    rel_lift = effect / abs(control_mean) if control_mean != 0 else 0.0

    return ExperimentResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        lift=effect,
        relative_lift=rel_lift,
        pvalue=pvalue,
        statistic=0.0,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=pvalue < 0.05,
        alpha=0.05,
        method="growthbook_bayesian",
        metric="migrated",
        control_n=control_n,
        treatment_n=treatment_n,
        power=float("nan"),
        effect_size=float("nan"),
    )


def _from_statsig(data: dict) -> ExperimentResult:
    """Convert Statsig frequentist output to ExperimentResult.

    Expected keys: p_value, effect_size, ci_lower, ci_upper,
    control_mean, treatment_mean, control_n, treatment_n.
    """
    _require_keys(
        data,
        ["p_value", "effect_size", "ci_lower", "ci_upper"],
        "statsig",
    )

    pvalue = float(data["p_value"])
    effect = float(data["effect_size"])
    ci_lower = float(data["ci_lower"])
    ci_upper = float(data["ci_upper"])
    control_mean = float(data.get("control_mean", 0.0))
    treatment_mean = float(data.get("treatment_mean", control_mean + effect))
    control_n = int(data.get("control_n", 0))
    treatment_n = int(data.get("treatment_n", 0))

    rel_lift = effect / abs(control_mean) if control_mean != 0 else 0.0

    return ExperimentResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        lift=effect,
        relative_lift=rel_lift,
        pvalue=pvalue,
        statistic=float(data.get("statistic", 0.0)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=pvalue < float(data.get("alpha", 0.05)),
        alpha=float(data.get("alpha", 0.05)),
        method="statsig_frequentist",
        metric="migrated",
        control_n=control_n,
        treatment_n=treatment_n,
        power=float("nan"),
        effect_size=float("nan"),
    )


def _from_generic(data: dict) -> ExperimentResult:
    """Convert a generic dict with effect/pvalue/ci to ExperimentResult.

    Expected keys: effect, pvalue, ci_lower, ci_upper.
    """
    _require_keys(data, ["effect", "pvalue", "ci_lower", "ci_upper"], "generic")

    effect = float(data["effect"])
    pvalue = float(data["pvalue"])
    ci_lower = float(data["ci_lower"])
    ci_upper = float(data["ci_upper"])
    control_mean = float(data.get("control_mean", 0.0))
    treatment_mean = float(data.get("treatment_mean", control_mean + effect))
    control_n = int(data.get("control_n", 0))
    treatment_n = int(data.get("treatment_n", 0))
    alpha = float(data.get("alpha", 0.05))

    rel_lift = effect / abs(control_mean) if control_mean != 0 else 0.0

    return ExperimentResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        lift=effect,
        relative_lift=rel_lift,
        pvalue=pvalue,
        statistic=float(data.get("statistic", 0.0)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=pvalue < alpha,
        alpha=alpha,
        method=str(data.get("method", "generic")),
        metric="migrated",
        control_n=control_n,
        treatment_n=treatment_n,
        power=float("nan"),
        effect_size=float("nan"),
    )
