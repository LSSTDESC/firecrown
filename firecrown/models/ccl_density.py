"""Critical and Mean Density  Mass Module - Implementation of the NumberDensity
abstract class for a critical and mean mass definition.
========================================
Class for mass definitions with Delta=200 times the critical/mean density.
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
import pyccl
from .number_density import NumberDensity

SUPPORTED_CRIT_DENS_FUNC_NAMES = {
    "Despali16": pyccl.halos.MassFuncDespali16,
    "Bocquet16": pyccl.halos.MassFuncBocquet16,
}
SUPPORTED_MEAN_DENS_FUNC_NAMES = {
    "Tinker08": pyccl.halos.MassFuncTinker08,
    "Tinker10": pyccl.halos.MassFuncTinker10,
}


class CCLDensity(NumberDensity):
    def __init__(
        self, density_func_definition, density_func_name, use_baryons=False
    ) -> None:
        """Default initialization for a base number density object."""
        super().__init__()
        self.use_baryons = use_baryons

        if density_func_name in SUPPORTED_CRIT_DENS_FUNC_NAMES:
            self.density_func_name = density_func_name
            self.pyccl_mass_func = SUPPORTED_CRIT_DENS_FUNC_NAMES[density_func_name]
            if density_func_definition == "mean":
                raise ValueError(
                    f"The number density function definition"
                    f"{density_func_definition}'%s' does not "
                    f"match the function type {density_func_name}'%s'"
                )
        elif density_func_name in SUPPORTED_MEAN_DENS_FUNC_NAMES:
            self.density_func_name = density_func_name
            self.pyccl_mass_func = SUPPORTED_MEAN_DENS_FUNC_NAMES[density_func_name]
            if density_func_definition == "critical":
                raise ValueError(
                    f"The number density function definition {density_func_definition}"
                    f"does not match the function type {density_func_name}'%s'"
                )
        else:
            raise ValueError(
                f"The number density function type {density_func_name}'%s' is not "
                f"supported!"
            )

    def compute_number_density(self, cosmo: pyccl.Cosmology, logm, z) -> float:
        """
        parameters

        cosmo : pyccl Cosmology
        logm: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.
        reuturn
        -------

        nm : float
            Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        a = 1.0 / (1.0 + z)
        mass = 10 ** (logm)
        hmd_200c = pyccl.halos.MassDef200c()
        if self.density_func_type == "Bocquet16":
            hmf_200c = self.pyccl_mass_func(
                cosmo,
                mass_def=hmd_200c,
                mass_def_strict=True,
                hydro=self.use_baryons,
            )
        else:
            hmf_200c = self.pyccl_mass_func(
                cosmo,
                mass_def=hmd_200c,
                mass_def_strict=True,
            )
        nm = hmf_200c.get_mass_function(cosmo, mass, a)
        return nm

    def compute_differential_comoving_volume(self, cosmo: pyccl.Cosmology, z) -> float:
        """
        parameters

        cosmo : pyccl Cosmology
        z : float
            Cluster Redshift.
        reuturn
        -------
        dv : float
            Differential Comoving Volume at z in units of Mpc^3 (comoving).
        """
        a = 1.0 / (1.0 + z)
        da = pyccl.background.angular_diameter_distance(cosmo, a)
        E = pyccl.background.h_over_h0(cosmo, a)
        dV = (
            ((1.0 + z) ** 2)
            * (da**2)
            * pyccl.physical_constants.CLIGHT_HMPC
            / cosmo["h"]
            / E
        )
        return dV
