"""write me"""
from abc import ABC, abstractmethod
from typing import Tuple
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand


class Integrator(ABC):
    """write me"""

    @abstractmethod
    def integrate(
        self,
        integrand: AbundanceIntegrand,
    ) -> float:
        """Integrate the integrand over the bounds and include extra_args to integral"""

    @abstractmethod
    def set_integration_bounds(
        self,
        cl_abundance: ClusterAbundance,
        z_proxy_limits: Tuple[float, float],
        mass_proxy_limits: Tuple[float, float],
    ) -> None:
        """Set the limits of integration and extra arguments for the integral"""
