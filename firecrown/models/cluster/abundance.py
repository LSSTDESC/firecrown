from typing import List
from pyccl.cosmology import Cosmology
import pyccl.background as bkg
import pyccl
from firecrown.models.cluster.kernel import Kernel, KernelType, ArgsMapper
import numpy as np
from firecrown.parameters import ParamsMap
from firecrown.integrator import Integrator


class ClusterAbundance(object):
    @property
    def sky_area(self) -> float:
        return self.sky_area_rad * (180.0 / np.pi) ** 2

    @sky_area.setter
    def sky_area(self, sky_area: float) -> None:
        self.sky_area_rad = sky_area * (np.pi / 180.0) ** 2

    @property
    def cosmo(self) -> Cosmology:
        return self._cosmo

    def __init__(
        self,
        min_mass: float,
        max_mass: float,
        min_z: float,
        max_z: float,
        halo_mass_function: pyccl.halos.MassFunc,
        sky_area: float,
        integrator: Integrator,
    ):
        self.kernels: List[Kernel] = []
        self.halo_mass_function = halo_mass_function
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_z = min_z
        self.max_z = max_z
        self.sky_area = sky_area
        self.integrator = integrator
        self._cosmo: Cosmology = None

    def add_kernel(self, kernel: Kernel):
        self.kernels.append(kernel)

    def update_ingredients(self, cosmo: Cosmology, params: ParamsMap):
        self._cosmo = cosmo
        for kernel in self.kernels:
            kernel.update(params)

    def comoving_volume(self, z) -> float:
        """Differential Comoving Volume at z.

        parameters
        :param ccl_cosmo: pyccl Cosmology
        :param z: Cluster Redshift.

        :return: Differential Comoving Volume at z in units of Mpc^3 (comoving).
        """
        scale_factor = 1.0 / (1.0 + z)
        angular_diam_dist = bkg.angular_diameter_distance(self.cosmo, scale_factor)

        h_over_h0 = bkg.h_over_h0(self.cosmo, scale_factor)
        dV = (
            pyccl.physical_constants.CLIGHT_HMPC
            * ((1.0 + z) ** 2)
            * (angular_diam_dist**2)
            / self.cosmo["h"]
            / h_over_h0
        )
        return dV * self.sky_area_rad

    def mass_function(self, mass: float, z: float) -> float:
        scale_factor = 1.0 / (1.0 + z)
        hmf = self.halo_mass_function(self.cosmo, 10**mass, scale_factor)
        return hmf

    def get_abundance_integrand(self):
        # Use the bounds mapping from the outer scope.

        def integrand(*int_args):
            args_map: ArgsMapper = int_args[-1]
            z = args_map.get_integral_bounds(int_args, KernelType.z)
            mass = args_map.get_integral_bounds(int_args, KernelType.mass)

            integrand = self.comoving_volume(z) * self.mass_function(mass, z)
            for kernel in self.kernels:
                integrand *= (
                    kernel.analytic_solution(int_args, args_map)
                    if kernel.has_analytic_sln
                    else kernel.distribution(int_args, args_map)
                )
            return integrand

        return integrand

    @property
    def analytic_kernels(self):
        return [x for x in self.kernels if x.has_analytic_sln]

    @property
    def dirac_delta_kernels(self):
        return [x for x in self.kernels if x.is_dirac_delta]

    @property
    def integrable_kernels(self):
        return [
            x for x in self.kernels if not x.is_dirac_delta and not x.has_analytic_sln
        ]

    def get_integration_bounds(self, z_proxy_limits, mass_proxy_limits):
        args_mapping = ArgsMapper()
        args_mapping.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}
        bounds_list = [(self.min_mass, self.max_mass), (self.min_z, self.max_z)]

        start_idx = len(args_mapping.integral_bounds.keys())

        for kernel in self.dirac_delta_kernels:
            # If any kernel is a dirac delta for z or M, just replace the
            # True limits with the proxy limits
            if kernel.kernel_type == KernelType.z_proxy:
                bounds_list[1] = z_proxy_limits
            elif kernel.kernel_type == KernelType.mass_proxy:
                bounds_list[0] = mass_proxy_limits

        for kernel in self.integrable_kernels:
            # If any kernel is not a dirac delta, integrate over the relevant limits
            if kernel.kernel_type == KernelType.z_proxy:
                args_mapping.integral_bounds[kernel.kernel_type.name] = start_idx
                bounds_list.append(z_proxy_limits)
            elif kernel.kernel_type == KernelType.mass_proxy:
                args_mapping.integral_bounds[kernel.kernel_type.name] = start_idx
                bounds_list.append(mass_proxy_limits)
            else:
                args_mapping.integral_bounds[kernel.kernel_type.name] = start_idx
                bounds_list.append(kernel.integral_bounds)
            start_idx += 1

        extra_args = []
        start_idx = 0
        for kernel in self.analytic_kernels:
            # Lastly, don't integrate any analyticly solved kernels, just solve them.
            if kernel.kernel_type == KernelType.z_proxy:
                args_mapping.extra_args[kernel.kernel_type.name] = start_idx
                extra_args.append(z_proxy_limits)
            elif kernel.kernel_type == KernelType.mass_proxy:
                args_mapping.extra_args[kernel.kernel_type.name] = start_idx
                extra_args.append(mass_proxy_limits)

            start_idx += 1

        return bounds_list, extra_args, args_mapping

    def compute(self, z_proxy_limits, mass_proxy_limits):
        bounds, extra_args, args_mapping = self.get_integration_bounds(
            z_proxy_limits, mass_proxy_limits
        )
        integrand = self.get_abundance_integrand()
        cc = self.integrator.integrate(integrand, bounds, args_mapping, extra_args)
        return cc
