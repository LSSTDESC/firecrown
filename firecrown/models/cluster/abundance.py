from typing import List
from pyccl.cosmology import Cosmology
import pyccl.background as bkg
import pyccl
from firecrown.models.cluster.kernel import Kernel, KernelType, ArgsMapping
import numpy as np
from firecrown.parameters import ParamsMap
import numpy.typing as npt

# TODO: 1. CCL recomputing and not caching - mass function
#       2. Integrator arguments issues
# * mass_function will ultimately be an NxN grid of values
#   This is probably being recomputed for every z, m bin
# * Power spectrum takes the heavy lifting, this will be computed for a
# single
#   cosmology.
# * Compute the grid of values for the mass function, then integrate
# Consider emulators for mass function


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
        self._hmf_cache = {}
        self._cosmo: Cosmology = None

    def add_kernel(self, kernel: Kernel):
        self.kernels.append(kernel)

    def update_ingredients(self, cosmo: Cosmology, params: ParamsMap):
        self._cosmo = cosmo
        self._hmf_cache = {}
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

    def mass_function(
        self, mass: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        scale_factor = 1.0 / (1.0 + z)

        return_vals = []
        for m, a in zip(mass, scale_factor):
            val = self._hmf_cache.get((m, a))
            if val is None:
                val = self.halo_mass_function(self.cosmo, 10**m, a)
                self._hmf_cache[(m, a)] = val
            return_vals.append(val)

        return return_vals

    def get_integrand(self, avg_mass=False, avg_redshift=False):
        def integrand(*int_args):
            args_map: ArgsMapping = int_args[-1]

            z = args_map.get_integral_bounds(int_args, KernelType.z)
            mass = args_map.get_integral_bounds(int_args, KernelType.mass)

            integrand = self.comoving_volume(z) * self.mass_function(mass, z)
            if avg_mass:
                integrand *= mass
            if avg_redshift:
                integrand *= z

            for kernel in self.kernels:
                # Think of overhead here, if its worth it
                integrand *= kernel.distribution(int_args, args_map)

            return integrand

        return integrand
