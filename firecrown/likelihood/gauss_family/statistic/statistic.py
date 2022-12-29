"""

Gaussian Family Statistic Module
================================

The Statistic class describing objects that implement methods to compute the
data and theory vectors for a GaussianFamily subclass.

"""

from __future__ import annotations
from typing import List
from dataclasses import dataclass
from abc import abstractmethod
import warnings

import numpy as np
import pyccl
import sacc

from ....updatable import Updatable
from .source.source import Systematic


@dataclass
class StatisticsResult:
    """This is the type returned by the `compute` method of any `Statistic`."""
    data: np.ndarray
    theory: np.ndarray

    def __post_init__(self):
        """Make sure the data and theory vectors are of the same shape."""
        assert self.data.shape == self.theory.shape

    def residuals(self):
        """Return the residuals -- the difference between data and theory."""
        return self.data - self.theory

    def __iter__(self):
        """Iterate through the data members. This is to allow automatic unpacking, as
        if the StatisticsResult were a tuple of (data, theory).

        This method is a temporary measure to help code migrate to the newer,
        safer interface for Statistic.compute()."""
        warnings.warn(
            "Iteration and tuple unpacking for StatisticsResult is "
            "deprecated.\nPlease use the StatisticsResult class accessors"
            ".data and .theory by name."
        )
        yield self.data
        yield self.theory


class Statistic(Updatable):
    """An abstract statistic class (e.g., two-point function, mass function, etc.)."""

    systematics: List[Systematic]
    sacc_inds: List[int]

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file."""

    @abstractmethod
    def compute(self, cosmo: pyccl.Cosmology) -> StatisticsResult:
        """Compute a statistic from sources, applying any systematics."""

        raise NotImplementedError("Method `compute` is not implemented!")
