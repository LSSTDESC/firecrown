"""Provides GaussFamily concrete types."""

from __future__ import annotations

from firecrown.likelihood.gaussian_base import ConstGaussianBase


class ConstGaussian(ConstGaussianBase):
    """A Gaussian log-likelihood with a constant covariance matrix."""

    # All functionality is inherited from ConstGaussianBase
