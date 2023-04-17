"""Cluster Redshift Module
abstract class to compute cluster redshift functions.
========================================
The implemented functions use PyCCL library as backend.
"""

from typing import final
from abc import abstractmethod

from ..updatable import Updatable
from ..parameters import ParamsMap


class ClusterRedshift(Updatable):
    """Cluster Redshift class."""

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
    def cluster_z_p(self, ccl_cosmo, logM, z, z_obs, z_obs_params):
        """Computes the logM proxy"""

    @abstractmethod
    def cluster_z_intp(self, ccl_cosmo, logM, z, z_obs, z_obs_params):
        """Computes the logM proxy"""
