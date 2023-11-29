"""The NumCosmo integrator module

This module holds the NumCosmo implementation of the integrator classes
"""
from typing import Tuple, Callable, Dict, Sequence, List, Optional
from enum import Enum
import numpy as np
import numpy.typing as npt
from numcosmo_py import Ncm
from firecrown.models.cluster.kernel import KernelType
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand
from firecrown.models.cluster.integrator.integrator import Integrator


class NumCosmoIntegralMethod(Enum):
    P = Ncm.IntegralNDMethod.P
    P_V = Ncm.IntegralNDMethod.P_V
    H = Ncm.IntegralNDMethod.H
    H_V = Ncm.IntegralNDMethod.H_V


class NumCosmoIntegrator(Integrator):
    """The NumCosmo implementation of the Integrator base class."""

    def __init__(
        self,
        method: Optional[NumCosmoIntegralMethod] = None,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-12,
    ) -> None:
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance
        self._method = method or NumCosmoIntegralMethod.P_V

        self.integral_args_lkp: Dict[KernelType, int] = self._default_integral_args()
        self.integral_bounds: List[Tuple[float, float]] = []

        self.z_proxy_limits: Tuple[float, float] = (-1.0, -1.0)
        self.mass_proxy_limits: Tuple[float, float] = (-1.0, -1.0)
        self.sky_area: float = 360**2

    def _integral_wrapper(
        self,
        integrand: AbundanceIntegrand,
    ) -> Callable[[npt.NDArray], Sequence[float]]:
        self._validate_integrand(integrand)

        # mypy strict issue: npt.NDArray[npt.NDArray[np.float64]] not supported
        def ncm_integrand(int_args: npt.NDArray) -> Sequence[float]:
            default = np.ones_like(int_args[0]) * -1.0
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
            ).tolist()
            assert isinstance(return_val, list)
            return return_val

        return ncm_integrand

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
        Ncm.cfg_init()
        ncm_integrand = self._integral_wrapper(integrand)
        int_nd = CountsIntegralND(len(self.integral_bounds), ncm_integrand)
        int_nd.set_method(self._method.value)
        int_nd.set_reltol(self._relative_tolerance)
        int_nd.set_abstol(self._absolute_tolerance)
        res = Ncm.Vector.new(1)
        err = Ncm.Vector.new(1)

        bl, bu = zip(*self.integral_bounds)
        int_nd.eval(Ncm.Vector.new_array(bl), Ncm.Vector.new_array(bu), res, err)
        return res.get(0)

    def _get_or_default(
        self,
        # mypy strict issue: npt.NDArray[npt.NDArray[np.float64]] not supported
        int_args: npt.NDArray,
        kernel_type: KernelType,
        default: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        try:
            return int_args[:, self.integral_args_lkp[kernel_type]]
        except KeyError:
            return default


class CountsIntegralND(Ncm.IntegralND):
    """Integral subclass used to compute the integrals using NumCosmo."""

    def __init__(
        self,
        dim: int,
        fun: Callable[[npt.NDArray], Sequence[float]],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fun = fun

    # pylint: disable-next=arguments-differ
    def do_get_dimensions(self) -> Tuple[int, int]:
        """Returns the dimensionality of the integral."""
        return self.dim, 1

    # pylint: disable-next=arguments-differ
    def do_integrand(
        self,
        x_vec: Ncm.Vector,
        dim: int,
        npoints: int,
        _fdim: int,
        fval_vec: Ncm.Vector,
    ) -> None:
        """Called by NumCosmo to evaluate the integrand."""
        x = np.array(x_vec.dup_array()).reshape(npoints, dim)
        fval_vec.set_array(self.fun(x))
