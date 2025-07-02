"""Module to compute the cluster excess surface mass density (delta sigma).

The galaxy cluster delta sigma integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

import numpy as np
import numpy.typing as npt
import pyccl
from scipy.stats import gamma
from scipy.integrate import simpson
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
import time
import clmm  # pylint: disable=import-error
from firecrown import parameters
from firecrown.models.cluster.abundance import ClusterAbundance


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
            self.cluster_conc = parameters.register_new_updatable_parameter(
                default_value=4.0
            )

    def delta_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        radius_center: np.float64,
        two_halo_term: bool = False,
        miscentering_frac: np.float64 = None,
        boost_factor: bool = False
    ) -> npt.NDArray[np.float64]:
        """Delta sigma for cprint(new_pred)lusters."""
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
        for log_m, redshift in zip(log_mass, z):
            # pylint: disable=protected-access
            conc_val = self._get_concentration(log_m, redshift)
            moo.set_concentration(conc_val)
            moo.set_mass(10**log_m)
            val = self._one_halo_contribution(moo, radius_center, redshift, miscentering_frac)
            if two_halo_term:
                val+= self._two_halo_contribution(moo, radius_center, redshift)
            if boost_factor:
                val = self._correct_with_boost_nfw(val, radius_center)
            return_vals.append(val)
        return np.asarray(return_vals, dtype=np.float64)


    def _one_halo_contribution(self, clmm_model: clmm.Modeling, radius_center, redshift, miscentering_frac = None, sigma_offset = 0.12) -> npt.NDArray[np.float64]:
        """Calculate the second halo contribution to the delta sigma."""
        # pylint: disable=protected-access
        #t1 = time.time()
        first_halo_right_centered = clmm_model.eval_excess_surface_density(radius_center, redshift)
        #t2 = time.time()
        #print(f'1halo term took {t2-t1}')
        if miscentering_frac is not None:
            integrator = NumCosmoIntegrator(relative_tolerance = 1e-2, absolute_tolerance = 1e-6,)
            def integration_func(int_args, extra_args):
                sigma = extra_args[0]
                r_mis_list = int_args[:, 0]
                #t3 = time.time()
                esd_vals = np.array([
                    clmm_model.eval_excess_surface_density(np.array([radius_center]), redshift, r_mis=r_mis)[0]
                    for r_mis in r_mis_list
                ])
                #t4 = time.time()
                #print(f'for {len(r_mis_list)} radiuses, misc it took {(t4 - t3) / len(r_mis_list)} per radius')
                gamma_vals = gamma.pdf(r_mis_list, a=2.0, scale=sigma)
                return esd_vals * gamma_vals
            integrator.integral_bounds = [(0.0, 25.0 * sigma_offset)]
            integrator.extra_args = np.array([sigma_offset])  ## From https://arxiv.org/pdf/2502.08444, we are using 0.12
            #t5 = time.time()
            miscentering_integral = integrator.integrate(integration_func)
            #t6 = time.time()
            #print(f'The miscentering integral took {t6 - t5}')
            return (1.0 - miscentering_frac) * first_halo_right_centered + miscentering_frac * miscentering_integral
        return first_halo_right_centered

    def _two_halo_contribution(self, clmm_model: clmm.Modeling, radius_center, redshift) -> npt.NDArray[np.float64]:
        """Calculate the second halo contribution to the delta sigma."""
        # pylint: disable=protected-access
        second_halo_right_centered = clmm_model.eval_excess_surface_density_2h(np.array([radius_center]), redshift)
        return second_halo_right_centered[0]

    def _get_concentration(self, log_m: float, redshift: float) -> float:
        """Determine the concentration for a halo."""
        if self.conc_parameter and self.cluster_conc is not None:
            return float(self.cluster_conc)

        conc_model = pyccl.halos.concentration.ConcentrationBhattacharya13(
            mass_def=self.halo_mass_function.mass_def
        )
        a = 1.0 / (1.0 + redshift)
        return conc_model._concentration(self._cosmo, 10.0**log_m, a)  # pylint: disable=protected-access
    
    def _correct_with_boost_nfw(self, profiles: npt.NDArray[np.float64], radius_list: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Determine the nfw boost factor and correct the shear profiles."""
        boost_factors = clmm.utils.compute_powerlaw_boost(radius_list, 1.0)
        #print(boost_factors, radius_list, profiles)
        corrected_profiles = clmm.utils.correct_with_boost_values(profiles, boost_factors)
        return corrected_profiles
