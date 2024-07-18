"""The module responsible for building a Delta_sigma calculation from mass and radius."""

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.background as bkg
from pyccl.cosmology import Cosmology
from firecrown.updatable import Updatable, UpdatableCollection


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
        cosmo_clmm = Cosmology()
        cosmo_clmm._init_from_cosmo(cosmo)
        self.moo = clmm.Modeling(massdef="mean", delta_mdef=200, halo_profile_model="nfw")
        self.moo.set_cosmo(cosmo_clmm)
        # assuming the same concentration for all masses. 
        # Not realistic, but avoid having to call a mass-concentration relation.
        self.moo.set_concentration(4) 

    def update_ingredients(self, cosmo: Cosmology) -> None:
        """Update the calculation with a new cosmology."""
        self._cosmo = cosmo
        self._hmf_cache = {}
        self.moo.set_cosmo(cosmo_clmm)

    def calculate_DS_from_Mass(
        self, 
        log10_mass:npt.NDArray[np.float64],
        redshift:npt.NDArray[np.float64],
        radii:npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
        
      
        mass = 10**log10_mass
        self.moo.set_mass(mass)
        profile = self.moo.eval_excess_surface_density(radii, redshift)

        assert isinstance(profile, np.ndarray)
        return profile

