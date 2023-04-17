"""Cluster Mass Module
abstract class to compute cluster mass function.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final
from abc import abstractmethod

from ..updatable import Updatable
from ..parameters import ParamsMap


class ClusterMass(Updatable):
    """Cluster Mass module."""

    @abstractmethod
    def _update_cluster_mass(self, params: ParamsMap):
        """Abstract method to update the ClusterMass from the given ParamsMap."""

    @abstractmethod
    def _reset_cluster_mass(self):
        """Abstract method to reset the ClusterMass."""

    @final
    def _update(self, params: ParamsMap):
        """Implementation of Updatable interface method `_update`."""

        self._update_cluster_mass(params)

    @final
    def _reset(self) -> None:
        """Implementation of the Updatable interface method `_reset`.

        This calls the abstract method `_reset_cluster_mass`, which must be implemented
        by all subclasses."""
        self._reset_cluster_mass()

    @abstractmethod  # pylint: disable-next=invalid-name
    def cluster_logM_p(self, logM, z, logM_obs):
        """Computes the logM proxytractmethod"""

    @abstractmethod  # pylint: disable-next=invalid-name
    def cluster_logM_intp(self, logM, z, logM_obs_lower, logM_obs_upper):
        """Computes logMintp"""
