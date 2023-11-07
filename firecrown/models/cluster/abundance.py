"""The module responsible for building the cluster abundance calculation.

The galaxy cluster abundance integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""
from typing import List, Callable, Optional, Dict, Tuple
import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.background as bkg
from pyccl.cosmology import Cosmology
from firecrown.models.cluster.kernel import Kernel
from firecrown.parameters import ParamsMap


AbundanceIntegrand = Callable[
    [
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        Tuple[float, float],
        Tuple[float, float],
    ],
    npt.NDArray[np.float64],
]


class ClusterAbundance:
    """The class that calculates the predicted number counts of galaxy clusters

    The abundance is a function of a specific cosmology, a mass and redshift range,
    an area on the sky, a halo mass function, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    abundance integrand.
    """

    @property
    def sky_area(self) -> float:
        """The sky area in square degrees

        The cluster number count prediction will be different depending on
        the area of the sky we are considering.  This value controls that.
        """
        return self.sky_area_rad * (180.0 / np.pi) ** 2

    @sky_area.setter
    def sky_area(self, sky_area: float) -> None:
        self.sky_area_rad = sky_area * (np.pi / 180.0) ** 2

    @property
    def cosmo(self) -> Cosmology:
        """The cosmology used to predict the cluster number count."""
        return self._cosmo

    @property
    def analytic_kernels(self) -> List[Kernel]:
        """The kernels in in the integrand which have an analytic solution."""
        return [x for x in self.kernels if x.has_analytic_sln]

    @property
    def dirac_delta_kernels(self) -> List[Kernel]:
        """The kernels in in the integrand which are dirac delta functions."""
        return [x for x in self.kernels if x.is_dirac_delta]

    @property
    def integrable_kernels(self) -> List[Kernel]:
        """The kernels in in the integrand which must be integrated."""
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
        """Add a kernel to the cluster abundance integrand"""
        self.kernels.append(kernel)

    def update_ingredients(
        self, cosmo: Cosmology, params: Optional[ParamsMap] = None
    ) -> None:
        """Update the cluster abundance calculation with a new cosmology."""
        self._cosmo = cosmo
        self._hmf_cache = {}
        if params is None:
            return

        for kernel in self.kernels:
            kernel.update(params)

    def comoving_volume(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The differential comoving volume at redshift z."""
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

    def mass_function(
        self, mass: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The mass function at z and mass."""
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
    ) -> AbundanceIntegrand:
        """Returns a callable that evaluates the complete integrand."""

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
