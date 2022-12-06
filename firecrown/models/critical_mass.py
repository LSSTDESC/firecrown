"""Critical Density  Mass Module - Implementation of the NumberDensity
abstract class for a critical mass definition.
========================================
Class for mass definitions with Delta=200 times the critical density.
"""
from __future__ import annotations
import pyccl
from .number_density import NumberDensity

SUPPORTED_DENS_FUNC_TYPES = {
    "Despali16": "d16",
    "Bocquet16": "b16",
}


class CriticalDensity(NumberDensity):
    def __init__(self, density_func_type, use_baryons=False) -> None:
        """Default initialization for a base number density object."""
        super().__init__()
        self.use_baryons = use_baryons
        if density_func_type in SUPPORTED_DENS_FUNC_TYPES:
            self.density_func_type = density_func_type
        else:
            raise ValueError(
                f"The number density function type {density_func_type}'%s' is not "
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
        if self.density_func_type == "Despali16":
            hmf_200c = pyccl.halos.MassFuncDespali16(
                cosmo, mass_def=hmd_200c, mass_def_strict=True
            )
        elif self.density_func_type == "Bocquet16":
            hmf_200c = pyccl.halos.MassFuncBocquet16(
                cosmo,
                mass_def=hmd_200c,
                mass_def_strict=True,
                hydro=self.use_baryons,
            )
        nm = hmf_200c.get_mass_function(cosmo, mass, a)
        return nm

    def compute_volume(self, cosmo: pyccl.Cosmology, z) -> float:
        """
        parameters

        cosmo : pyccl Cosmology
        z : float
            Cluster Redshift.
        reuturn
        -------
        dv : float
            Volume Density pdf at z in units of Mpc^3 (comoving).
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
