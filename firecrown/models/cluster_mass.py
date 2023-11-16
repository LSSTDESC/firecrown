"""Cluster Mass Module
Abstract class to compute cluster mass function.


The implemented functions use the :mod:`pyccl` library.
"""

from __future__ import annotations
from typing import final, List, Tuple, Optional
from abc import abstractmethod

import numpy as np
import sacc

from ..updatable import Updatable
from ..parameters import ParamsMap


class ClusterMassArgument:
    """Cluster Mass argument class."""

    def __init__(self, logMl: float, logMu: float):
        self.logMl: float = logMl
        self.logMu: float = logMu
        self.logM: Optional[float] = None
        self.dirac_delta: bool = False

        if logMl > logMu:
            raise ValueError("logMl must be smaller than logMu")
        if logMl == logMu:
            self.dirac_delta = True
            self.logM = logMl

    def is_dirac_delta(self) -> bool:
        """Check if the argument is a dirac delta."""

        return self.dirac_delta

    def get_logM(self) -> float:
        """Return the logM value if the argument is a dirac delta."""

        if self.logM is not None:
            return self.logM
        raise ValueError("Argument is not a Dirac delta")

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the argument."""

    @abstractmethod
    def get_logM_bounds(self) -> Tuple[float, float]:
        """Return the bounds of the cluster mass argument."""

    @abstractmethod
    def get_proxy_bounds(self) -> List[Tuple[float, float]]:
        """Return the bounds of the cluster mass proxy argument."""

    @abstractmethod
    def p(self, logM: float, z: float, *proxy_args) -> float:
        """Return the probability of the argument."""


class ClusterMass(Updatable):
    """Cluster Mass module."""

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc):
        """Abstract method to read the data for this source from the SACC file."""

    def _update_cluster_mass(self, params: ParamsMap):
        """Method to update the ClusterMass from the given ParamsMap.
        Subclasses that need to do more than update their contained
        :class:`Updatable` instance variables should implement this method."""

    @final
    def _update(self, params: ParamsMap):
        """Implementation of Updatable interface method `_update`."""

        self._update_cluster_mass(params)

    @abstractmethod
    def gen_bins_by_array(self, logM_obs_bins: np.ndarray) -> List[ClusterMassArgument]:
        """Generate bins by an array of bin edges."""

    @abstractmethod
    def gen_bin_from_tracer(self, tracer: sacc.BaseTracer) -> ClusterMassArgument:
        """Return the bin for the given tracer."""
