from typing import List, Dict
from pyccl.cosmology import Cosmology
import pyccl.background as bkg
import pyccl
from scipy.integrate import nquad

from firecrown.models.kernel import Kernel
import numpy as np


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
        halo_mass_function: pyccl.halos.MassFunc,
        sky_area_rad: float = 4 * np.pi**2,
    ):
        self.kernels: List[Kernel] = []
        self.halo_mass_function = halo_mass_function
        self.sky_area_rad = sky_area_rad
        self._cosmo: Cosmology = None

    def add_kernel(self, kernel: Kernel):
        self.kernels.append(kernel)

    def update_ingredients(self, cosmo: Cosmology):
        self._cosmo = cosmo

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

    def get_abundance_integrand(self):
        def integrand(args: List[float], index_lookup: Dict[str, float]):
            z = args[index_lookup["z"]]
            mass = args[index_lookup["mass"]]

            integrand = self.comoving_volume(z) * self.mass_function(mass, z)
            for kernel in self.kernels:
                integrand *= kernel.distribution(args, index_lookup)
            return integrand

        return integrand

    def get_abundance_bounds(self):
        index_lookup = {}
        bounds_list = []
        # Keep a separate index instead of using enumerate so we don't increment
        # when we skip a dirac delta function
        idx = 0
        for kernel in self.kernels:
            if kernel.is_dirac_delta:
                continue
            index_lookup[kernel.kernel_type.name] = idx
            bounds_list.append(kernel.integral_bounds)
            idx += 1

        return bounds_list, index_lookup

    def compute(self):
        # Args = x0...xn, index_lkp = t0,...tm
        # The function to be integrated.
        # Has arguments of x0, ... xn, t0, ... tm, where integration is carried out
        # over x0, ... xn, which must be floats. Where t0, ... tm are extra arguments
        # passed in args. Function signature should be
        # func(x0, x1, ..., xn, t0, t1, ..., tm). Integration is carried out in order.
        # That is, integration over x0 is the innermost integral, and xn is the
        # outermost.
        integrand = self.get_abundance_integrand()
        bounds_list, index_lookup = self.get_abundance_bounds()
        return nquad(
            # Function of the form f(x0, x1, ..., xn, t0, t1, ..., tm)
            integrand,
            # Each element of ranges must be a sequence of 2 numbers
            # e.g. ranges[0] = (a,b) for f(x0, x1)
            ranges=bounds_list,
            # t0,...tn
            args=(index_lookup),
            opts={"epsabs": 1e-12, "epsrel": 1e-4},
        )

    # Firecrown specific
    def _process_args(self, args):
        x = np.array(args[0:-5])
        index_map, arg, ccl_cosmo, mass_arg, redshift_arg = args[-5:]
        arg[index_map] = x
        redshift_start_index = 2 + redshift_arg.dim

        logM, z = arg[0:2]
        proxy_z = arg[2:redshift_start_index]
        proxy_m = arg[redshift_start_index:]

        return logM, z, proxy_z, proxy_m, ccl_cosmo, mass_arg, redshift_arg
