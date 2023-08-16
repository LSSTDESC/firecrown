from pyccl.cosmology import Cosmology
import pyccl.background as bkg
import pyccl


class ClusterAbundance(object):
    def __init__(
        self,
        cosmo: Cosmology,
        halo_mass_function: pyccl.halos.MassFunc,
        sky_area: float,
    ):
        self.cosmo = cosmo
        self.halo_mass_function = halo_mass_function
        self.sky_area = sky_area

    def update_ingredients(self, cosmo: Cosmology):
        self.cosmo = cosmo
        # Set cosmology dependent cluster abundance values.
        pass

    def comoving_volume(self, z) -> float:
        """Differential Comoving Volume at z.

        parameters
        :param ccl_cosmo: pyccl Cosmology
        :param z: Cluster Redshift.

        :return: Differential Comoving Volume at z in units of Mpc^3 (comoving).
        """
        scale_factor = 1.0 / (1.0 + z)
        da = bkg.angular_diameter_distance(self.cosmo, scale_factor)

        h_over_h0 = bkg.h_over_h0(self.cosmo, scale_factor)
        dV = (
            ((1.0 + z) ** 2)
            * (da**2)
            * pyccl.physical_constants.CLIGHT_HMPC
            / self.cosmo["h"]
            / h_over_h0
        )
        return dV * self.sky_area_rad

    def mf_d2N_dz_dlnM_completeness(self, logM: float, z: float) -> float:
        """
        Computes the mass function at z and logM.

        :param ccl_cosmo: pyccl Cosmology
        :param logM: Cluster mass given by log10(M) where M is in units of
            M_sun (comoving).
        :param z: Cluster Redshift.
        :return: Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        mass_function = self.mass_funcion(self.cosmo, logM, z)
        comoving_volume = self.comoving_volume(self.cosmo, z)
        completeness = self._cluster_abundance_compute_completeness(logM, z)

        return mass_function * comoving_volume * completeness

    def mass_funcion(self, logM: float, z: float) -> float:
        """
        Computes the mass function at z and logM.

        :param ccl_cosmo: pyccl Cosmology
        :param logM: Cluster mass given by log10(M) where M is in units of
            M_sun (comoving).
        :param z: Cluster Redshift.
        :return: Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        scale_factor = 1.0 / (1.0 + z)
        mass = 10 ** (logM)
        nm = self.halo_mass_function(self.cosmo, mass, scale_factor)
        return nm
