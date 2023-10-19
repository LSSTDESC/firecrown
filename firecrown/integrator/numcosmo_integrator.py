from numcosmo_py import Ncm
from typing import Tuple
import numpy as np
from firecrown.models.cluster.kernel import ArgReader, KernelType
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.integrator import Integrator


class NumCosmoArgReader(ArgReader):
    def __init__(self):
        super().__init__()
        self.integral_bounds = dict()
        self.extra_args = dict()

        self.integral_bounds_idx = 0
        self.extra_args_idx = 1

    def get_integral_bounds(self, int_args, kernel_type: KernelType):
        bounds_values = int_args[self.integral_bounds_idx]
        return bounds_values[:, self.integral_bounds[kernel_type.name]]

    def get_extra_args(self, int_args, kernel_type: KernelType):
        extra_values = int_args[self.extra_args_idx]
        return extra_values[self.extra_args[kernel_type.name]]


class NumCosmoIntegrator(Integrator):
    def __init__(self, relative_tolerance=1e-4, absolute_tolerance=1e-12):
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance
        self.arg_reader = NumCosmoArgReader()

    def get_integration_bounds(
        self, cl_abundance: ClusterAbundance, z_proxy_limits, mass_proxy_limits
    ):
        self.arg_reader.integral_bounds = {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

        integral_bounds = [
            (cl_abundance.min_mass, cl_abundance.max_mass),
            (cl_abundance.min_z, cl_abundance.max_z),
        ]

        # If any kernel is a dirac delta for z or M, just replace the
        # true limits with the proxy limits
        for kernel in cl_abundance.dirac_delta_kernels:
            if kernel.kernel_type == KernelType.z_proxy:
                integral_bounds[1] = z_proxy_limits
            elif kernel.kernel_type == KernelType.mass_proxy:
                integral_bounds[0] = mass_proxy_limits

        # If any kernel is not a dirac delta, integrate over the relevant limits
        mapping_idx = len(self.arg_reader.integral_bounds.keys())
        for kernel in cl_abundance.integrable_kernels:
            self.arg_reader.integral_bounds[kernel.kernel_type.name] = mapping_idx
            mapping_idx += 1

            if kernel.kernel_type == KernelType.z_proxy:
                integral_bounds.append(z_proxy_limits)
            elif kernel.kernel_type == KernelType.mass_proxy:
                integral_bounds.append(mass_proxy_limits)

            if kernel.integral_bounds is not None:
                integral_bounds.append(kernel.integral_bounds)

        # Lastly, don't integrate any kernels with an analytic solution
        # This means we pass in their limits as extra arguments to the integrator
        extra_args = []
        self.arg_reader.extra_args = {}
        for i, kernel in enumerate(cl_abundance.analytic_kernels):
            self.arg_reader.extra_args[kernel.kernel_type.name] = i

            if kernel.kernel_type == KernelType.z_proxy:
                extra_args.append(z_proxy_limits)
            elif kernel.kernel_type == KernelType.mass_proxy:
                extra_args.append(mass_proxy_limits)

            if kernel.integral_bounds is not None:
                extra_args.append(kernel.integral_bounds)

        return integral_bounds, extra_args

    def integrate(self, integrand, bounds, extra_args):
        Ncm.cfg_init()
        int_nd = CountsIntegralND(
            len(bounds),
            integrand,
            extra_args,
            self.arg_reader,
        )
        int_nd.set_method(Ncm.IntegralNDMethod.P_V)
        int_nd.set_reltol(self._relative_tolerance)
        int_nd.set_abstol(self._absolute_tolerance)
        res = Ncm.Vector.new(1)
        err = Ncm.Vector.new(1)

        bl, bu = zip(*bounds)
        int_nd.eval(Ncm.Vector.new_array(bl), Ncm.Vector.new_array(bu), res, err)
        return res.get(0)


class CountsIntegralND(Ncm.IntegralND):
    """Integral subclass used by the ClusterAbundance
    to compute the integrals using numcosmo."""

    def __init__(self, dim, fun, *args):
        super().__init__()
        self.dim = dim
        self.fun = fun
        self.args = args

    # pylint: disable-next=arguments-differ
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
        fval_vec.set_array(self.fun(x, *self.args))
