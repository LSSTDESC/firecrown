"""Cluster Mass Module
abstract class to compute cluster mass function.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import final, List, Tuple, Optional
from abc import abstractmethod

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
        elif logMl == logMu:
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

    @abstractmethod
    def get_args(self) -> List[ClusterMassArgument]:
        """Return the argument generator of the cluster mass function."""
