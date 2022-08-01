"""

Gaussian Family Statistic Module
================================

The Statistic class describing objects that implement methods to compute the
data and theory vectors for a GaussianFamily subclass.

"""

from __future__ import annotations
from typing import List, Tuple
from abc import abstractmethod
import numpy as np
import pyccl
import sacc

from ....updatable import Updatable
from .source.source import Systematic


class Statistic(Updatable):
    """A statistic (e.g., two-point function, mass function, etc.)."""

    systematics: List[Systematic]
    sacc_inds: List[int]

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file."""
        pass

    @abstractmethod
    def compute(self, cosmo: pyccl.Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a statistic from sources, applying any systematics."""

        raise NotImplementedError("Method `compute` is not implemented!")
