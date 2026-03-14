"""Variance reduction techniques for A/B testing."""

from splita.variance.cupac import CUPAC
from splita.variance.cuped import CUPED
from splita.variance.outliers import OutlierHandler

__all__ = ["CUPAC", "CUPED", "OutlierHandler"]
