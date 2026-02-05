"""Module to compute the cluster excess surface mass density (delta sigma).

The galaxy cluster delta sigma integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

import clmm  # pylint: disable=import-error
import numpy as np
import numpy.typing as npt
import pyccl

from firecrown.updatable import register_new_updatable_parameter
from firecrown.models.cluster._abundance import ClusterAbundance


class ClusterDeltaSigma(ClusterAbundance):
    """The class that calculates the predicted delta sigma of galaxy clusters.

    The excess density surface mass density is a function of a specific cosmology,
    a mass and redshift range, an area on the sky, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    shear integrand.
    """

    def __init__(
        self,
        mass_interval: tuple[float, float],
        z_interval: tuple[float, float],
        halo_mass_function: pyccl.halos.MassFunc,
        conc_parameter: bool = False,
    ) -> None:
        super().__init__(mass_interval, z_interval, halo_mass_function)
        self.conc_parameter = conc_parameter
        if conc_parameter:
            self.cluster_conc = register_new_updatable_parameter(default_value=4.0)

    def delta_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        radius_center: float,
    ) -> npt.NDArray[np.float64]:
        """Delta sigma for clusters."""
        cosmo_clmm = clmm.Cosmology()
        # pylint: disable=protected-access
        cosmo_clmm._init_from_cosmo(self._cosmo)
        mass_def = self.halo_mass_function.mass_def
        mass_type = mass_def.rho_type
        if mass_type == "matter":
            mass_type = "mean"
        moo = clmm.Modeling(
            massdef=mass_type,
            delta_mdef=mass_def.Delta,
            halo_profile_model="nfw",
        )
        moo.set_cosmo(cosmo_clmm)
        return_vals = []
        if not self.conc_parameter:
            conc = pyccl.halos.concentration.ConcentrationBhattacharya13(
                mass_def=mass_def
            )
            for log_m, redshift in zip(log_mass, z, strict=False):
                a = 1.0 / (1.0 + redshift)
                conc_val = conc(self._cosmo, 10**log_m, a)
                moo.set_concentration(conc_val)
                moo.set_mass(10**log_m)
                val = moo.eval_excess_surface_density(radius_center, redshift)
                return_vals.append(val)
        else:
            conc_val = self.cluster_conc
            moo.set_concentration(conc_val)
            for log_m, redshift in zip(log_mass, z, strict=False):
                moo.set_concentration(conc_val)
                moo.set_mass(10**log_m)
                val = moo.eval_excess_surface_density(radius_center, redshift)
                return_vals.append(val)
        return np.asarray(return_vals, dtype=np.float64)
