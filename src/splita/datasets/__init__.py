"""Sample datasets for splita examples and tutorials.

Each ``load_*`` function returns a deterministic (seeded) dataset dict
with NumPy arrays and a human-readable description of the scenario.

Examples
--------
>>> from splita.datasets import load_ecommerce
>>> data = load_ecommerce()
>>> data["description"][:20]
'E-commerce A/B test:'
"""

from splita.datasets.generators import (
    load_ecommerce,
    load_marketplace,
    load_mobile_app,
    load_subscription,
)

__all__ = [
    "load_ecommerce",
    "load_marketplace",
    "load_mobile_app",
    "load_subscription",
]
