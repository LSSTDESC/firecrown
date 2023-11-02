from scipy.integrate import nquad
from typing import Callable, Dict, Tuple
import numpy.typing as npt
from firecrown.models.cluster.integrator.integrator import Integrator
from firecrown.models.cluster.kernel import KernelType
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand
import numpy as np


class ScipyIntegrator(Integrator):
    def __init__(
        self, relative_tolerance: float = 1e-4, absolute_tolerance: float = 1e-12
    ) -> None:
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

        self.integral_args_lkp: Dict[KernelType, int] = self._default_integral_args()

        self.z_proxy_limits: Tuple[float, float] = (-1.0, -1.0)
        self.mass_proxy_limits: Tuple[float, float] = (-1.0, -1.0)

    def _default_integral_args(self) -> Dict[KernelType, int]:
        lkp: Dict[KernelType, int] = dict()
        lkp[KernelType.mass] = 0
        lkp[KernelType.z] = 1
        return lkp

    def _integral_wrapper(
        self,
        integrand: AbundanceIntegrand,
    ) -> Callable[..., float]:
        def scipy_integrand(*int_args: float) -> float:
            default = -1.0

            mass = self._get_or_default(int_args, KernelType.mass, default)
            z = self._get_or_default(int_args, KernelType.z, default)
            mass_proxy = self._get_or_default(int_args, KernelType.mass_proxy, default)
            z_proxy = self._get_or_default(int_args, KernelType.z_proxy, default)

            return_val = integrand(
                mass,
                z,
                mass_proxy,
                z_proxy,
                self.mass_proxy_limits,
                self.z_proxy_limits,
            )

            return return_val[0]

        return scipy_integrand

    def set_integration_bounds(
        self,
        cl_abundance: ClusterAbundance,
        z_proxy_limits: Tuple[float, float],
        mass_proxy_limits: Tuple[float, float],
    ) -> None:
        # pdb.set_trace()
        self.integral_args_lkp = self._default_integral_args()
        self.integral_bounds = [
            (cl_abundance.min_mass, cl_abundance.max_mass),
            (cl_abundance.min_z, cl_abundance.max_z),
        ]

        self.mass_proxy_limits = mass_proxy_limits
        self.z_proxy_limits = z_proxy_limits

        for kernel in cl_abundance.dirac_delta_kernels:
            if kernel.kernel_type == KernelType.z_proxy:
                self.integral_bounds[1] = z_proxy_limits

            elif kernel.kernel_type == KernelType.mass_proxy:
                self.integral_bounds[0] = mass_proxy_limits

        for kernel in cl_abundance.integrable_kernels:
            idx = len(self.integral_bounds)

            if kernel.kernel_type == KernelType.z_proxy:
                self.integral_bounds.append(z_proxy_limits)
                self.integral_args_lkp[KernelType.z_proxy] = idx

            elif kernel.kernel_type == KernelType.mass_proxy:
                self.integral_bounds.append(mass_proxy_limits)
                self.integral_args_lkp[KernelType.mass_proxy] = idx
        return

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
