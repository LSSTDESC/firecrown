"""Cluster Mass True Module

Class to compute cluster mass functions with no proxy,
i.e., assuming we have the true masses of the clusters.

"""

from typing import final, List, Tuple

import numpy as np
from .. import sacc_support
from ..sacc_support import sacc


from ..parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from .cluster_mass import ClusterMass, ClusterMassArgument


class ClusterMassTrue(ClusterMass):
    """Cluster Mass class."""

    @final
    def _update_cluster_mass(self, params: ParamsMap):
        """Method to update the ClusterMassTrue from the given ParamsMap."""

    @final
    def _reset_cluster_mass(self):
        """Method to reset the ClusterMassTrue."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def read(self, sacc_data: sacc.Sacc):
        """Method to read the data for this source from the SACC file."""

    def gen_bins_by_array(self, logM_bins: np.ndarray) -> List[ClusterMassArgument]:
        """Generate the bins by an array of bin edges."""

        if len(logM_bins) < 2:
            raise ValueError("logM_bins must have at least two elements")

        # itertools.pairwise is only available in Python 3.10
        # using zip instead
        return [
            ClusterMassTrueArgument(lower, upper)
            for lower, upper in zip(logM_bins[:-1], logM_bins[1:])
        ]

    def point_arg(self, logM: float) -> ClusterMassArgument:
        """Return the argument for the given mass."""

        return ClusterMassTrueArgument(logM, logM)

    def gen_bin_from_tracer(self, tracer: sacc.BaseTracer) -> ClusterMassArgument:
        """Return the argument for the given tracer."""

        if not isinstance(tracer, sacc_support.BinLogMTracer):
            raise ValueError("Tracer must be a BinLogMTracer")

        return ClusterMassTrueArgument(tracer.lower, tracer.upper)


class ClusterMassTrueArgument(ClusterMassArgument):
    """Cluster mass true argument class."""

    @property
    def dim(self) -> int:
        """Return the dimension of the argument."""
        return 0

    def get_logM_bounds(self) -> Tuple[float, float]:
        """Return the bounds of the cluster mass argument."""
        return (self.logMl, self.logMu)

    def get_proxy_bounds(self) -> List[Tuple[float, float]]:
        """Return the bounds of the cluster mass proxy argument."""
        return []

    def p(self, logM: float, z: float, *args) -> float:
        """Return the probability of the argument."""
        return 1.0
