"""The module responsible for building the cluster abundance calculation.

The galaxy cluster abundance integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.background as bkg
from pyccl.cosmology import Cosmology
from firecrown.updatable import Updatable, UpdatableCollection


class ClusterAbundance(Updatable):
    """The class that calculates the predicted number counts of galaxy clusters.

    The abundance is a function of a specific cosmology, a mass and redshift range,
    an area on the sky, a halo mass function, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    abundance integrand.
    """

    @property
    def cosmo(self) -> Cosmology | None:
        """The cosmology used to predict the cluster number count."""
        return self._cosmo

    def __init__(
        self,
        mass_interval: tuple[float, float],
        z_interval: tuple[float, float],
        halo_mass_function: pyccl.halos.MassFunc,
    ) -> None:
        super().__init__()
        self.kernels: UpdatableCollection = UpdatableCollection()
        self.halo_mass_function = halo_mass_function
        self.min_mass = mass_interval[0]
        self.max_mass = mass_interval[1]
        self.min_z = z_interval[0]
        self.max_z = z_interval[1]
        self._hmf_cache: dict[tuple[float, float], float] = {}
        self._cosmo: Cosmology | None = None

    def update_ingredients(self, cosmo: Cosmology) -> None:
        """Update the cluster abundance calculation with a new cosmology."""
        self._cosmo = cosmo
        self._hmf_cache = {}

    def comoving_volume(
        self, z: npt.NDArray[np.float64], sky_area: float = 0
    ) -> npt.NDArray[np.float64]:
        """The differential comoving volume given area sky_area at redshift z.

        :param sky_area: The area of the survey on the sky in square degrees.
        """
        assert self.cosmo is not None
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

        sky_area_rad = sky_area * (np.pi / 180.0) ** 2

        return np.array(dV * sky_area_rad, dtype=np.float64)

    def mass_function(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """The mass function at z and mass."""
        scale_factor = 1.0 / (1.0 + z)
        return_vals = []

        for m, a in zip(mass.astype(float), scale_factor.astype(float)):
            val = self._hmf_cache.get((m, a))
            if val is None:
                val = self.halo_mass_function(self.cosmo, 10**m, a)
                self._hmf_cache[(m, a)] = val
            return_vals.append(val)

        return np.asarray(return_vals, dtype=np.float64)
