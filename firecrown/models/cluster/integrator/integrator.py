"""The integrator module

This module holds the classes that define the interface required to
integrate an assembled cluster abundance.
"""
import inspect
from abc import ABC, abstractmethod
from typing import Tuple, Dict, get_args, List
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand
from firecrown.models.cluster.kernel import KernelType


class Integrator(ABC):
    """The integrator base class

    This class acts as an adapter around an integration library, and must provides
    a specific set of methods to be used to integrate a cluster abundance integral."""

    def __init__(self) -> None:
        self.z_proxy_limits: Tuple[float, float] = (-1.0, -1.0)
        self.mass_proxy_limits: Tuple[float, float] = (-1.0, -1.0)
        self.sky_area: float = 360**2
        self.integral_bounds: List[Tuple[float, float]] = []
        self.integral_args_lkp: Dict[KernelType, int] = self._default_integral_args()

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
        sky_area: float,
        z_proxy_limits: Tuple[float, float],
        mass_proxy_limits: Tuple[float, float],
    ) -> None:
        """Sets the limits of integration used by the integration library.

        This method uses the mass and redshift proxy arguments, along with
        the cluster abundance argument to construct the limits of integration
        to be used in evaluating the cluster abundance."""

    def _default_integral_args(self) -> Dict[KernelType, int]:
        lkp: Dict[KernelType, int] = {}
        lkp[KernelType.MASS] = 0
        lkp[KernelType.Z] = 1
        return lkp

    def _validate_integrand(self, integrand: AbundanceIntegrand) -> None:
        expected_args, expected_return = get_args(AbundanceIntegrand)

        signature = inspect.signature(integrand)
        params = signature.parameters.values()
        param_types = [param.annotation for param in params]

        assert len(params) == len(expected_args)

        assert param_types == list(expected_args)

        assert signature.return_annotation == expected_return
