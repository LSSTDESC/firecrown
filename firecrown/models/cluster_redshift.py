"""Cluster Redshift Module
abstract class to compute cluster redshift functions.
========================================
The implemented functions use PyCCL library as backend.
"""

from typing import final, List, Tuple, Optional
from abc import abstractmethod

import numpy as np

from ..sacc_support import sacc
from ..updatable import Updatable
from ..parameters import ParamsMap


class ClusterRedshiftArgument:
    """Cluster Redshift argument class."""

    def __init__(self, zl: float, zu: float):
        self.zl: float = zl
        self.zu: float = zu
        self.z: Optional[float] = None
        self.dirac_delta: bool = False

        if zl > zu:
            raise ValueError("zl must be smaller than zu")
        if zl == zu:
            self.dirac_delta = True
            self.z = zl

    def is_dirac_delta(self) -> bool:
        """Check if the argument is a dirac delta."""

        return self.dirac_delta

    def get_z(self) -> float:
        """Return the z value if the argument is a dirac delta."""

        if self.z is not None:
            return self.z
        raise ValueError("Argument is not a Dirac delta")

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the argument."""

    @abstractmethod
    def get_z_bounds(self) -> Tuple[float, float]:
        """Return the bounds of the cluster redshift argument."""

    @abstractmethod
    def get_proxy_bounds(self) -> List[Tuple[float, float]]:
        """Return the bounds of the cluster redshift proxy argument."""

    @abstractmethod
    def p(self, logM: float, z: float, *args) -> float:
        """Return the probability of the argument."""


class ClusterRedshift(Updatable):
    """Cluster Redshift class."""

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc):
        """Abstract method to read the data for this source from the SACC file."""

    @abstractmethod
    def _update_cluster_redshift(self, params: ParamsMap):
        """Abstract method to update the ClusterRedshift from the given ParamsMap."""

    @abstractmethod
    def _reset_cluster_redshift(self):
        """Abstract method to reset the ClusterRedshift."""

    @final
    def _update(self, params: ParamsMap):
        """Implementation of Updatable interface method `_update`."""

        self._update_cluster_redshift(params)

    @final
    def _reset(self) -> None:
        """Implementation of the Updatable interface method `_reset`.

        This calls the abstract method `_reset_cluster_redshift`, which must be
        implemented by all subclasses."""
        self._reset_cluster_redshift()

    @abstractmethod
    def gen_bins_by_array(self, z_bins: np.ndarray) -> List[ClusterRedshiftArgument]:
        """Generate the bins by an array of bin edges."""

    @abstractmethod
    def gen_bin_from_tracer(self, tracer: sacc.BaseTracer) -> ClusterRedshiftArgument:
        """Return the bin for the given tracer."""
