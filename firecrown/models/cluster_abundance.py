from typing import List
from pyccl.cosmology import Cosmology
import pyccl.background as bkg
import pyccl

from firecrown.models.kernel import Kernel
import numpy as np


class ClusterAbundance(object):
    @property
    def sky_area(self) -> float:
        return self.sky_area_rad * (180.0 / np.pi) ** 2

    @sky_area.setter
    def sky_area(self, sky_area: float) -> None:
        self.sky_area_rad = sky_area * (np.pi / 180.0) ** 2

    def __init__(
        self,
        halo_mass_function: pyccl.halos.MassFunc,
    ):
        self.kernels: List[Kernel] = []
        self.cosmo = None
        self.halo_mass_function = halo_mass_function

    def add_kernel(self, kernel: Kernel):
        self.kernels.append(kernel)

    def update_ingredients(self, cosmo: Cosmology):
        self.cosmo = cosmo

    def comoving_volume(self, z) -> float:
        """Differential Comoving Volume at z.

        parameters
        :param ccl_cosmo: pyccl Cosmology
        :param z: Cluster Redshift.

        :return: Differential Comoving Volume at z in units of Mpc^3 (comoving).
        """
        scale_factor = 1.0 / (1.0 + z)
        da = bkg.angular_diameter_distance(self.cosmo, scale_factor)

        h_over_h0 = bkg.h_over_h0(self.cosmo, scale_factor)
        dV = (
            ((1.0 + z) ** 2)
            * (da**2)
            * pyccl.physical_constants.CLIGHT_HMPC
            / self.cosmo["h"]
            / h_over_h0
        )
        return dV * self.sky_area_rad

    def mass_function(self, mass: float, z: float) -> float:
        scale_factor = 1.0 / (1.0 + z)
        hmf = self.halo_mass_function(self.cosmo, 10**mass, scale_factor)
        return hmf

    def build_integrand(self, mass, z):
        integrand = self.comoving_volume(z) * self.mass_function(mass, z)
        for kernel in self.kernels:
            integrand *= kernel.distribution(mass, z)

    def compute(self):
        integrand = self.build_integrand()
