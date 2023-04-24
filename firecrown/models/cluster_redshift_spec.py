"""Cluster Redshift Spectrocopy Module

Class to compute cluster redshift spectroscopy functions.

"""

from typing import final, List, Tuple
import itertools

import sacc
import numpy as np

from ..parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from .cluster_redshift import ClusterRedshift, ClusterRedshiftArgument


class ClusterRedshiftSpecArgument(ClusterRedshiftArgument):
    """Cluster Redshift spectroscopy argument class."""

    @property
    def dim(self) -> int:
        """Return the dimension of the argument."""
        return 0

    def get_z_bounds(self) -> Tuple[float, float]:
        """Return the bounds of the cluster redshift argument."""
        return (self.zl, self.zu)

    def get_proxy_bounds(self) -> List[Tuple[float, float]]:
        """Return the bounds of the cluster redshift proxy argument."""
        return []

    def p(self, logM: float, z: float, *args) -> float:
        """Return the probability of the argument."""
        return 1.0


class ClusterRedshiftSpec(ClusterRedshift):
    """Cluster Redshift class."""

    @final
    def _update_cluster_redshift(self, params: ParamsMap):
        """Method to update the ClusterRedshiftSpec from the given ParamsMap."""

    @final
    def _reset_cluster_redshift(self):
        """Method to reset the ClusterRedshiftSpec."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def read(self, sacc_data: sacc.Sacc):
        """Method to read the data for this source from the SACC file."""

    def gen_bins_by_array(self, z_bins: np.ndarray) -> List[ClusterRedshiftArgument]:
        """Generate the bins by an array of bin edges."""

        if len(z_bins) < 2:
            raise ValueError("z_bins must have at least two elements")

        return [
            ClusterRedshiftSpecArgument(z_lower, z_upper)
            for z_lower, z_upper in itertools.pairwise(z_bins)
        ]

    def point_arg(self, z: float) -> ClusterRedshiftArgument:
        """Return the argument for the given redshift."""

        return ClusterRedshiftSpecArgument(z, z)
