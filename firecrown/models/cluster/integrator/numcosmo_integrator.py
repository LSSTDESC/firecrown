from numcosmo_py import Ncm
from typing import Tuple, Callable, Dict, Sequence
import numpy as np
import numpy.typing as npt
from firecrown.models.cluster.kernel import KernelType
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand
from firecrown.models.cluster.integrator.integrator import Integrator


class NumCosmoIntegrator(Integrator):
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
    ) -> Callable[[npt.NDArray], Sequence[float]]:
        # mypy strict issue: npt.NDArray[npt.NDArray[np.float64]] not supported
        def ncm_integrand(int_args: npt.NDArray) -> Sequence[float]:
            default = np.ones_like(int_args[0]) * -1.0

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
            ).tolist()
            assert isinstance(return_val, list)
            return return_val

        return ncm_integrand

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
        Ncm.cfg_init()
        ncm_integrand = self._integral_wrapper(integrand)
        int_nd = CountsIntegralND(len(self.integral_bounds), ncm_integrand)
        int_nd.set_method(Ncm.IntegralNDMethod.P_V)
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
    """Integral subclass used by the ClusterAbundance
    to compute the integrals using numcosmo."""

    def __init__(
        self,
        dim: int,
        fun: Callable[[npt.NDArray], Sequence[float]],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fun = fun

    def do_get_dimensions(self) -> Tuple[int, int]:
        """Get number of dimensions."""
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
        """Integrand function."""
        x = np.array(x_vec.dup_array()).reshape(npoints, dim)
        fval_vec.set_array(self.fun(x))
