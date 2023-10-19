from scipy.integrate import nquad

from firecrown.integrator import Integrator
from firecrown.models.cluster.kernel import ArgReader, KernelType
from firecrown.models.cluster.abundance import ClusterAbundance
import numpy as np


class ScipyIntegrator(Integrator):
    def __init__(self, relative_tolerance=1e-4, absolute_tolerance=1e-12):
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance
        self.arg_reader = ScipyArgReader()

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

        for kernel in cl_abundance.analytic_kernels:
            self.arg_reader.extra_args[kernel.kernel_type.name] = mapping_idx
            mapping_idx += 1
            if kernel.kernel_type == KernelType.z_proxy:
                extra_args.append(z_proxy_limits)
            elif kernel.kernel_type == KernelType.mass_proxy:
                extra_args.append(mass_proxy_limits)

            if kernel.integral_bounds is not None:
                extra_args.append(kernel.integral_bounds)

        return integral_bounds, extra_args

    def integrate(self, integrand, bounds, extra_args):
        val = nquad(
            integrand,
            ranges=bounds,
            args=(*extra_args, self.arg_reader),
            opts={
                "epsabs": self._absolute_tolerance,
                "epsrel": self._relative_tolerance,
            },
        )[0]

        return val


class ScipyArgReader(ArgReader):
    def __init__(self):
        super().__init__()
        self.integral_bounds = dict()
        self.extra_args = dict()
        self.integral_bounds_idx = 0

    def get_integral_bounds(self, int_args, kernel_type: KernelType):
        return np.array(int_args[self.integral_bounds[kernel_type.name]])

    def get_extra_args(self, int_args, kernel_type: KernelType):
        return int_args[self.extra_args[kernel_type.name]]
