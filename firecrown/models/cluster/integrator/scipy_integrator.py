"""The SciPy integrator module

This module holds the scipy implementation of the integrator classes
"""
from typing import Callable, Dict, Tuple
import numpy as np
import numpy.typing as npt
from scipy.integrate import nquad
from firecrown.models.cluster.integrator.integrator import Integrator
from firecrown.models.cluster.kernel import KernelType
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand


class ScipyIntegrator(Integrator):
    """The scipy implementation of the Integrator base class using nquad."""

    def __init__(
        self, relative_tolerance: float = 1e-4, absolute_tolerance: float = 1e-12
    ) -> None:
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

        self.integral_args_lkp: Dict[KernelType, int] = self._default_integral_args()

    def _integral_wrapper(
        self,
        integrand: AbundanceIntegrand,
    ) -> Callable[..., float]:
        self._validate_integrand(integrand)

        def scipy_integrand(*int_args: float) -> float:
            default = -1.0
            # pylint: disable=R0801
            mass = self._get_or_default(int_args, KernelType.MASS, default)
            z = self._get_or_default(int_args, KernelType.Z, default)
            mass_proxy = self._get_or_default(int_args, KernelType.MASS_PROXY, default)
            z_proxy = self._get_or_default(int_args, KernelType.Z_PROXY, default)

            return_val = integrand(
                mass,
                z,
                self.sky_area,
                mass_proxy,
                z_proxy,
                self.mass_proxy_limits,
                self.z_proxy_limits,
            )[0]
            assert isinstance(return_val, float)
            return return_val

        return scipy_integrand

    def set_integration_bounds(
        self,
        cl_abundance: ClusterAbundance,
        sky_area: float,
        z_proxy_limits: Tuple[float, float],
        mass_proxy_limits: Tuple[float, float],
    ) -> None:
        # pylint: disable=R0801
        self.integral_args_lkp = self._default_integral_args()
        self.integral_bounds = [
            (cl_abundance.min_mass, cl_abundance.max_mass),
            (cl_abundance.min_z, cl_abundance.max_z),
        ]

        self.mass_proxy_limits = mass_proxy_limits
        self.z_proxy_limits = z_proxy_limits
        self.sky_area = sky_area

        for kernel in cl_abundance.dirac_delta_kernels:
            if kernel.kernel_type == KernelType.Z_PROXY:
                self.integral_bounds[1] = z_proxy_limits

            elif kernel.kernel_type == KernelType.MASS_PROXY:
                self.integral_bounds[0] = mass_proxy_limits

        for kernel in cl_abundance.integrable_kernels:
            idx = len(self.integral_bounds)

            if kernel.kernel_type == KernelType.Z_PROXY:
                self.integral_bounds.append(z_proxy_limits)
                self.integral_args_lkp[KernelType.Z_PROXY] = idx

            elif kernel.kernel_type == KernelType.MASS_PROXY:
                self.integral_bounds.append(mass_proxy_limits)
                self.integral_args_lkp[KernelType.MASS_PROXY] = idx

    def integrate(
        self,
        integrand: AbundanceIntegrand,
    ) -> float:
        scipy_integrand = self._integral_wrapper(integrand)
        val = nquad(
            scipy_integrand,
            ranges=self.integral_bounds,
            opts={
                "epsabs": self._absolute_tolerance,
                "epsrel": self._relative_tolerance,
            },
        )[0]
        assert isinstance(val, float)
        return val

    def _get_or_default(
        self,
        int_args: Tuple[float, ...],
        kernel_type: KernelType,
        default: float,
    ) -> npt.NDArray[np.float64]:
        try:
            return np.atleast_1d(int_args[self.integral_args_lkp[kernel_type]])
        except KeyError:
            return np.atleast_1d(default)
