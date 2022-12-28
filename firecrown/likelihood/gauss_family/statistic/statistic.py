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
        return self.data - self.theory


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
