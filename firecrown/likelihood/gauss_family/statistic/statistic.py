"""

Gaussian Family Statistic Module
================================

The Statistic class describing objects that implement methods to compute the
data and theory vectors for a GaussianFamily subclass.

"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pyccl
import sacc

from ....updatable import Updatable
from .source.source import Systematic


class Statistic(Updatable):
    """An abstract statistic class (e.g., two-point function, mass function, etc.)."""

    systematics: List[Systematic]
    sacc_indices: List[int]

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file."""

    def get_data_vector(self) -> np.ndarray:
        """Gets the statistic data vector."""
        raise NotImplementedError("Method `get_data_vector` is not implemented!")

    def compute_theory_vector(self, cosmo: pyccl.Cosmology) -> np.ndarray:
        """Compute a statistic from sources, applying any systematics."""
        raise NotImplementedError("Method `compute_theory_vector` is not implemented!")

    def compute(self, cosmo: pyccl.Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a statistic from sources, applying any systematics."""

        raise NotImplementedError("Method `compute` is not implemented!")
