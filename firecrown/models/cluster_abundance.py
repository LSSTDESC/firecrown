from typing import List, Dict
from pyccl.cosmology import Cosmology
import pyccl.background as bkg
import pyccl
from scipy.integrate import nquad
from itertools import product
from firecrown.models.kernel import Kernel, KernelType
import numpy as np
import pdb
from firecrown.parameters import ParamsMap


class ClusterAbundance(object):
    _absolute_tolerance = 1e-12
    _relative_tolerance = 1e-4

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
    ):
        self.kernels: List[Kernel] = []
        self.halo_mass_function = halo_mass_function
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_z = min_z
        self.max_z = max_z
        self.sky_area = sky_area
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
            ((1.0 + z) ** 2)
            * (angular_diam_dist**2)
            * pyccl.physical_constants.CLIGHT_HMPC
            / self.cosmo["h"]
            / h_over_h0
        )
        return dV * self.sky_area_rad

    def mass_function(self, mass: float, z: float) -> float:
        scale_factor = 1.0 / (1.0 + z)
        hmf = self.halo_mass_function(self.cosmo, 10**mass, scale_factor)
        return hmf

    def get_abundance_integrand(self, bounds_map, args_map):
        def integrand(*args):
            z = args[bounds_map["z"]]
            mass = args[bounds_map["mass"]]
            integrand = self.comoving_volume(z) * self.mass_function(mass, z)
            for kernel in self.kernels:
                if kernel.has_analytic_sln:
                    integrand *= kernel.analytic_solution(args, bounds_map, args_map)
                else:
                    integrand *= kernel.distribution(args, bounds_map)
            return integrand

        return integrand

    def get_integration_bounds(self):
        index_lookup = {"mass": 0, "z": 1}
        bounds_list = [[(self.min_mass, self.max_mass)], [(self.min_z, self.max_z)]]
        idx = 2
        for kernel in self.kernels:
            if kernel.integral_bounds is None or kernel.has_analytic_sln:
                continue

            if not kernel.is_dirac_delta:
                index_lookup[kernel.kernel_type.name] = idx
                bounds_list.append(kernel.integral_bounds)
                idx += 1
                continue

            # If either z or m has a proxy, and its a dirac delta, just replace the
            # limits
            if kernel.kernel_type == KernelType.z_proxy:
                bounds_list[index_lookup["z"]] = kernel.integral_bounds
            elif kernel.kernel_type == KernelType.mass_proxy:
                bounds_list[index_lookup["mass"]] = kernel.integral_bounds

        bounds_by_bin = list(product(*bounds_list))

        return bounds_by_bin, index_lookup

    def get_analytic_args(self):
        idx = 0
        index_lookup = {}
        args = []
        for kernel in self.kernels:
            if not kernel.has_analytic_sln:
                continue
            index_lookup[kernel.kernel_type.name] = idx
            args.append(kernel.integral_bounds)
            idx += 1

        return args, index_lookup

    def compute(self):
        bounds_by_bin, bounds_map = self.get_integration_bounds()
        analytic_args, args_map = self.get_analytic_args()
        integrand = self.get_abundance_integrand(bounds_map, args_map)

        cluster_counts = []
        print(bounds_by_bin)
        for bounds in bounds_by_bin:
            for analytic_sln in analytic_args:
                for analytic_bounds in analytic_sln:
                    cc = nquad(
                        integrand,
                        ranges=bounds,
                        args=analytic_bounds,
                        opts={
                            "epsabs": self._absolute_tolerance,
                            "epsrel": self._relative_tolerance,
                        },
                    )[0]
                    print(cc)
                    cluster_counts.append(cc)

        print(cluster_counts)
        return cluster_counts
