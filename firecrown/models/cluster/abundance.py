from typing import List, Callable, Optional, Dict, Tuple
from pyccl.cosmology import Cosmology
import pyccl.background as bkg
import pyccl
from firecrown.models.cluster.kernel import Kernel
import numpy as np
from firecrown.parameters import ParamsMap
import numpy.typing as npt

# from functools import singledispatchmethod
# import pdb


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
    def analytic_kernels(self) -> List[Kernel]:
        return [x for x in self.kernels if x.has_analytic_sln]

    @property
    def dirac_delta_kernels(self) -> List[Kernel]:
        return [x for x in self.kernels if x.is_dirac_delta]

    @property
    def integrable_kernels(self) -> List[Kernel]:
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
    ) -> None:
        self.kernels: List[Kernel] = []
        self.halo_mass_function = halo_mass_function
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_z = min_z
        self.max_z = max_z
        self.sky_area = sky_area
        self._hmf_cache: Dict[Tuple[float, float], float] = {}
        self._cosmo: Cosmology = None

    def add_kernel(self, kernel: Kernel) -> None:
        self.kernels.append(kernel)

    def update_ingredients(
        self, cosmo: Cosmology, params: Optional[ParamsMap] = None
    ) -> None:
        self._cosmo = cosmo
        self._hmf_cache = {}
        if params is None:
            return

        for kernel in self.kernels:
            kernel.update(params)

    # @singledispatchmethod
    # def comoving_volume(
    #     self, z: Union[float, npt.NDArray[np.float64]]
    # ) -> Union[float, npt.NDArray[np.float64]]:
    #     """Differential Comoving Volume at z.

    #     parameters
    #     :param ccl_cosmo: pyccl Cosmology
    #     :param z: Cluster Redshift.

    #     :return: Differential Comoving Volume at z in units of Mpc^3 (comoving).
    #     """
    #     raise ValueError("Unsupported type for z:", type(z))

    # @comoving_volume.register(np.ndarray)
    def comoving_volume(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        scale_factor = 1.0 / (1.0 + z)
        angular_diam_dist = bkg.angular_diameter_distance(self.cosmo, scale_factor)
        h_over_h0 = bkg.h_over_h0(self.cosmo, scale_factor)

        dV = (
            pyccl.physical_constants.CLIGHT_HMPC
            * (angular_diam_dist**2)
            * ((1.0 + z) ** 2)
            / (self.cosmo["h"] * h_over_h0)
        )
        assert isinstance(dV, np.ndarray)
        return dV * self.sky_area_rad

    # @comoving_volume.register(float)
    # def _(self, z: float) -> float:
    #     z_arr = np.atleast_1d(z)
    #     vol = self.comoving_volume(z_arr)
    #     assert isinstance(vol, np.ndarray)
    #     return vol.item()

    # @singledispatchmethod
    # def mass_function(
    #     self,
    #     mass: Union[float, npt.NDArray[np.float64]],
    #     z: Union[float, npt.NDArray[np.float64]],
    # ) -> Union[float, npt.NDArray[np.float64]]:
    #     """Halo Mass Function at mass and z."""
    #   raise ValueError("Unsupported type for either mass or z:", type(mass), type(z))

    # @mass_function.register(float)
    # def _(self, mass: float, z: float) -> float:
    #     z_arr = np.atleast_1d(z)
    #     mass_arr = np.atleast_1d(mass)
    #     mf = self.mass_function(mass_arr, z_arr)
    #     assert isinstance(mf, np.ndarray)
    #     return mf.item()

    # @mass_function.register(np.ndarray)
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

        return np.asarray(return_vals, dtype=np.float64)

    def get_integrand(
        self, avg_mass: bool = False, avg_redshift: bool = False
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            Tuple[float, float],
            Tuple[float, float],
        ],
        npt.NDArray[np.float64],
    ]:
        def integrand(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy: npt.NDArray[np.float64],
            z_proxy: npt.NDArray[np.float64],
            mass_proxy_limits: Tuple[float, float],
            z_proxy_limits: Tuple[float, float],
        ) -> npt.NDArray[np.float64]:
            integrand = self.comoving_volume(z) * self.mass_function(mass, z)

            if avg_mass:
                integrand *= mass
            if avg_redshift:
                integrand *= z

            for kernel in self.kernels:
                integrand *= kernel.distribution(
                    mass, z, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
                )
            return integrand

        return integrand