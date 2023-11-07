"""The integrator module

This module holds the classes that define the interface required to
integrate an assembled cluster abundance.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand


class Integrator(ABC):
    """The integrator base class

    This class acts as an adapter around an integration library, and must provides
    a specific set of methods to be used to integrate a cluster abundance integral."""

    @abstractmethod
    def integrate(
        self,
        integrand: AbundanceIntegrand,
    ) -> float:
        """Call this method to integrate the provided integrand argument."""

    @abstractmethod
    def set_integration_bounds(
        self,
        cl_abundance: ClusterAbundance,
        z_proxy_limits: Tuple[float, float],
        mass_proxy_limits: Tuple[float, float],
    ) -> None:
        """Sets the limits of integration used by the integration library.

        This method uses the mass and redshift proxy arguments, along with
        the cluster abundance argument to construct the limits of integration
        to be used in evaluating the cluster abundance."""
