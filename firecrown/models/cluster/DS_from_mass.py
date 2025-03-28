"""The module builds a Delta_sigma calculation from mass, redshift and radius."""

import os
import numpy as np
import numpy.typing as npt
from pyccl.cosmology import Cosmology
from firecrown.updatable import Updatable, UpdatableCollection
import clmm


class DS_from_mass(Updatable):
    """The class that calculates the predicted number counts of galaxy clusters.

    The abundance is a function of a specific cosmology, a mass and redshift range,
    an area on the sky, a halo mass function, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    abundance integrand.
    """

    @property
    def cosmo(self) -> Cosmology:
        """The cosmology used to predict the cluster number count."""
        return self._cosmo

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.kernels: UpdatableCollection = UpdatableCollection()
        self._cosmo: Cosmology = None

        # Setting things up for CLMM
        os.environ["CLMM_MODELING_BACKEND"] = "ccl"
        self.cosmo_clmm = clmm.Cosmology()
        self.moo = clmm.Modeling(
            massdef="mean", delta_mdef=200, halo_profile_model="nfw"
        )

    def update_ingredients(self, cosmo: Cosmology) -> None:
        """Update the calculation with a new cosmology."""
        self._cosmo = cosmo

        # Setting things up for CLMM
        self.cosmo_clmm._init_from_cosmo(cosmo)
        self.moo.set_cosmo(self.cosmo_clmm)
        # assuming the same concentration for all masses.
        # Not realistic, but avoid having to call a mass-concentration relation.
        self.moo.set_concentration(4)

    def calculate_DS_from_Mass(
        self,
        log10_mass: npt.NDArray[np.float64],
        redshift: npt.NDArray[np.float64],
        radii: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate DeltaSigma for a given log10_mass, redshift, and radii.

        Input:
        log10_mass: numpy.NDArray[np.float64]; cluster mass value in log10.
        redshift: numpy.NDArray[np.float64]; redshift values.
        radii: numpy.NDArray[np.float64]; radius to compute DeltaSigma at.

        Output:
        profile: numpy NDArray[np.float64]; delta_sigma profile corresponding to
        the log10_mass, redshift and radii
        """
        mass = 10**log10_mass
        self.moo.set_mass(mass)
        profile = self.moo.eval_excess_surface_density(radii, redshift)

        return profile
