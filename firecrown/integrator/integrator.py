from abc import ABC, abstractmethod
from firecrown.models.cluster.kernel import ArgReader
from firecrown.models.cluster.abundance import ClusterAbundance


class Integrator(ABC):
    arg_reader: ArgReader

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def integrate(self, integrand, bounds, extra_args):
        """Integrate the integrand over the bounds and include extra_args to integral"""

    @abstractmethod
    def get_integration_bounds(
        self, cl_abundance: ClusterAbundance, z_proxy_limits, mass_proxy_limits
    ):
        """Extract the limits of integration and extra arguments for the integral"""
