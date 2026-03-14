"""Bayesian stopping rules for bandits (planned for v0.2.0)."""


class BayesianStopping:
    """Bayesian stopping rules for multi-armed bandits.

    .. note::
        This class is planned for v0.2.0 and is not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BayesianStopping is planned for splita v0.2.0. "
            "Use ThompsonSampler with stopping_rule='expected_loss' instead."
        )
