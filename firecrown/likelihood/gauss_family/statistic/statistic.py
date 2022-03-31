"""The Statistic class describing ...

"""

from __future__ import annotations
from typing import Dict, Optional
from abc import ABC, abstractmethod
from typing import final
import numpy as np
import pyccl
import sacc

class Statistic(ABC):
    """A statistic (e.g., two-point function, mass function, etc.).

    Parameters
    ----------
    sources : list of str
        A list of the sources needed to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.
    """

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        pass

    @final
    def update_params(self, params):
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.update_params(params)
        self._update_params(params)
    
    @abstractmethod
    def _update_params(self, params):
        pass

    @abstractmethod
    def compute(
        self,
        cosmo: pyccl.Cosmology,
        params: Dict[str, float],
    ) -> (np.ndarray, np.ndarray):
        """Compute a statistic from sources, applying any systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        """
        raise NotImplementedError("Method `compute` is not implemented!")
