"""The module for building the cluster excess surface mass density (delta sigma) calculation.

The galaxy cluster delta sigma integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

import numpy as np
import numpy.typing as npt
import pyccl

import clmm
from pyccl.cosmology import Cosmology
from firecrown.updatable import Updatable, UpdatableCollection
from firecrown import parameters


class ClusterDeltaSigma(Updatable):
    """The class that calculates the predicted delta sigma of galaxy clusters.

    The excess density surface mass density is a function of a specific cosmology,
    a mass and redshift range, an area on the sky, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    shear integrand.
    """

    @property
    def cosmo(self) -> Cosmology:
        """The cosmology used to predict the cluster number count."""
        return self._cosmo

    def __init__(
        self,
        min_mass: float,
        max_mass: float,
        min_z: float,
        max_z: float,
        halo_mass_function: pyccl.halos.MassFunc,
        conc_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.kernels: UpdatableCollection = UpdatableCollection()
        self.halo_mass_function = halo_mass_function
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_z = min_z
        self.max_z = max_z
        self._hmf_cache: dict[tuple[float, float], float] = {}
        self._cosmo: Cosmology = None
        self.conc_parameter = conc_parameter
        if conc_parameter:
            self.cluster_conc = parameters.register_new_updatable_parameter(
                default_value=4.0
            )

    def update_ingredients(self, cosmo: Cosmology) -> None:
        """Update the cluster abundance calculation with a new cosmology."""
        self._cosmo = cosmo
        self._hmf_cache = {}

    def delta_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        radius_center: float,
    ) -> npt.NDArray[np.float64]:
        """Delta sigma for clusters."""
        cosmo_clmm = clmm.Cosmology()
        cosmo_clmm._init_from_cosmo(self._cosmo)
        mass_def = self.halo_mass_function.mass_def
        moo = clmm.Modeling(
            massdef=mass_def.rho_type,
            delta_mdef=mass_def.Delta,
            halo_profile_model="nfw",
        )
        moo.set_cosmo(cosmo_clmm)
        return_vals = []
        if not self.conc_parameter:
            conc = pyccl.halos.concentration.ConcentrationBhattacharya13(
                mass_def=mass_def
            )
            for log_m, redshift in zip(log_mass, z):
                a = 1.0 / (1.0 + redshift)
                conc_val = conc._concentration(self._cosmo, 10**log_m, a)
                moo.set_concentration(conc_val)
                moo.set_mass(10**log_m)
                val = moo.eval_excess_surface_density(radius_center, redshift)
                return_vals.append(np.log10(val))
        else:
            conc_val = self.cluster_conc
            moo.set_concentration(conc_val)
            for log_m, redshift in zip(log_mass, z):
                moo.set_concentration(conc_val)
                moo.set_mass(10**log_m)
                val = moo.eval_excess_surface_density(radius_center, redshift)
                return_vals.append(np.log10(val))
        return np.asarray(return_vals, dtype=np.float64)
