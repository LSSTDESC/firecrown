r"""Cluster Abundance Module
abstract class to compute cluster abundance.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import Optional, Any, Dict, List, Tuple, final
import itertools

import numpy as np
import scipy.integrate
import pyccl as ccl

from ..updatable import Updatable
from ..parameters import ParamsMap, RequiredParameters, DerivedParameterCollection
from .cluster_mass import ClusterMass
from .cluster_redshift import ClusterRedshift


class ClusterAbundance(Updatable):
    r"""Cluster Abundance class"""

    def __init__(
        self,
        halo_mass_definition: ccl.halos.MassDef,
        halo_mass_function_name: str,
        halo_mass_function_args: Dict[str, Any],
        cluster_mass: ClusterMass,
        cluster_redshift: ClusterRedshift,
        sky_area: float,
        use_completness: bool = False,
        use_purity: bool = False,
    ):
        """Initialize the ClusterAbundance class."""
        super().__init__()
        self.cluster_m = cluster_mass
        self.cluster_z = cluster_redshift
        self.sky_area = sky_area
        self.sky_area_rad = self.sky_area * (np.pi / 180.0) ** 2
        self.halo_mass_definition = halo_mass_definition
        self.halo_mass_function_name = halo_mass_function_name
        self.halo_mass_function_args = halo_mass_function_args
        self.halo_mass_function: Optional[ccl.halos.MassFunc] = None
        self.use_purity = use_purity

        if use_completness:
            self.base_mf_d2N_dz_dlnM = self.mf_d2N_dz_dlnM_completeness
        else:
            self.base_mf_d2N_dz_dlnM = self.mf_d2N_dz_dlnM

    @final
    def _update(self, params: ParamsMap):
        """Implementation of Updatable interface method `_update`."""
        self.cluster_m.update(params)
        self.cluster_z.update(params)

    @final
    def _reset(self) -> None:
        """Implementation of the Updatable interface method `_reset`."""
        self.cluster_m.reset()
        self.cluster_z.reset()
        self.halo_mass_function = None

    @final
    def _required_parameters(self) -> RequiredParameters:
        return (
            self.cluster_m.required_parameters() + self.cluster_z.required_parameters()
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])
        derived_parameters = (
            derived_parameters + self.cluster_m.get_derived_parameters()
        )
        derived_parameters = (
            derived_parameters + self.cluster_z.get_derived_parameters()
        )
        return derived_parameters

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.

        :param sacc_data: The data in the sacc format.
        """

        self.cluster_m.read(sacc_data)
        self.cluster_z.read(sacc_data)

    def dV_dz(self, ccl_cosmo: ccl.Cosmology, z) -> float:
        """Differential Comoving Volume at z.

        parameters
        :param ccl_cosmo: pyccl Cosmology
        :param z: Cluster Redshift.

        :return: Differential Comoving Volume at z in units of Mpc^3 (comoving).
        """
        a = 1.0 / (1.0 + z)
        da = ccl.background.angular_diameter_distance(ccl_cosmo, a)
        E = ccl.background.h_over_h0(ccl_cosmo, a)
        dV = (
            ((1.0 + z) ** 2)
            * (da**2)
            * ccl.physical_constants.CLIGHT_HMPC
            / ccl_cosmo["h"]
            / E
        )
        return dV * self.sky_area_rad

    def mf_d2N_dV_dlnM(self, ccl_cosmo: ccl.Cosmology, logM: float, z: float) -> float:
        """
        Computes the mass function at z and logM.

        :param ccl_cosmo: pyccl Cosmology
        :param logM: Cluster mass given by log10(M) where M is in units of
            M_sun (comoving).
        :param z: Cluster Redshift.
        :return: Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        a = 1.0 / (1.0 + z)
        mass = 10 ** (logM)
        if self.halo_mass_function is None:
            self.halo_mass_function = ccl.halos.MassFunc.from_name(
                self.halo_mass_function_name
            )(ccl_cosmo, **self.halo_mass_function_args)
        nm = self.halo_mass_function.get_mass_function(ccl_cosmo, mass, a)
        return nm

    def mf_d2N_dz_dlnM(self, ccl_cosmo: ccl.Cosmology, logM: float, z: float) -> float:
        """
        Computes the mass function at z and logM.

        :param ccl_cosmo: pyccl Cosmology
        :param logM: Cluster mass given by log10(M) where M is in units of
            M_sun (comoving).
        :param z: Cluster Redshift.
        :return: Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        d2N_dV_dlnM = self.mf_d2N_dV_dlnM(ccl_cosmo, logM, z)
        dV_dz = self.dV_dz(ccl_cosmo, z)

        return d2N_dV_dlnM * dV_dz

    def mf_d2N_dz_dlnM_completeness(
        self, ccl_cosmo: ccl.Cosmology, logM: float, z: float
    ) -> float:
        """
        Computes the mass function at z and logM.

        :param ccl_cosmo: pyccl Cosmology
        :param logM: Cluster mass given by log10(M) where M is in units of
            M_sun (comoving).
        :param z: Cluster Redshift.
        :return: Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        d2N_dV_dlnM = self.mf_d2N_dV_dlnM(ccl_cosmo, logM, z)
        dV_dz = self.dV_dz(ccl_cosmo, z)
        completeness = self._cluster_abundance_compute_completeness(logM, z)

        return d2N_dV_dlnM * dV_dz * completeness

    def _cluster_abundance_compute_purity(self, logM_obs, z):
        ln_r = np.log(10**logM_obs)
        a_nc = np.log(10) * 0.8612
        b_nc = np.log(10) * 0.3527
        a_rc = 2.2183
        b_rc = -0.6592
        nc = a_nc + b_nc * (1.0 + z)
        ln_rc = a_rc + b_rc * (1.0 + z)
        purity = (ln_r / ln_rc) ** nc / ((ln_r / ln_rc) ** nc + 1.0)
        return purity

    def _cluster_abundance_compute_completeness(self, logM, z):
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc * (1.0 + z)
        nc = a_nc + b_nc * (1.0 + z)
        C = (logM / log_mc) ** nc / ((logM / log_mc) ** nc + 1.0)
        return C

    def _process_args(self, args):
        x = np.array(args[0:-5])
        index_map = args[-5]
        arg = args[-4]
        ccl_cosmo = args[-3]
        mass_arg = args[-2]
        redshift_arg = args[-1]

        arg[index_map] = x

        logM, z = arg[0:2]
        proxy_z = arg[2: 2 + redshift_arg.dim]
        proxy_m = arg[2 + redshift_arg.dim:]

        return logM, z, proxy_z, proxy_m, ccl_cosmo, mass_arg, redshift_arg

    # Generic integrand for the cluster abundance
    # The arg array always has the following structure:
    # [logM, z, proxy_z, proxy_m]
    # where proxy_z and proxy_m are the proxy parameters for the redshift and
    # mass arguments respectively.
    # The index_map array is used to map the proxy parameters to the correct
    # position in the arg array.
    def _compute_integrand(self, *args):
        (
            logM,
            z,
            proxy_z,
            proxy_m,
            ccl_cosmo,
            mass_arg,
            redshift_arg,
        ) = self._process_args(args)

        return (
            self.base_mf_d2N_dz_dlnM(ccl_cosmo, logM, z)
            * mass_arg.p(logM, z, *proxy_m)
            * redshift_arg.p(logM, z, *proxy_z)
        )

    def _compute_integrand_mean_logM(self, *args):
        (
            logM,
            z,
            proxy_z,
            proxy_m,
            ccl_cosmo,
            mass_arg,
            redshift_arg,
        ) = self._process_args(args)

        return (
            self.base_mf_d2N_dz_dlnM(ccl_cosmo, logM, z)
            * mass_arg.p(logM, z, *proxy_m)
            * redshift_arg.p(logM, z, *proxy_z)
            * logM
        )

    def _compute_any(self, integrand, ccl_cosmo: ccl.Cosmology) -> np.ndarray:
        """Computes the cluster abundance in all bins defined
        by the cluster mass and redshift proxy bins."""

        mass_args = self.cluster_m.get_args()
        redshift_args = self.cluster_z.get_args()

        # Compute the cluster abundance in each argument.
        if len(mass_args) != len(redshift_args):
            raise ValueError(
                "The number of mass and redshift arguments must be the same."
            )

        res = []
        for mass_arg, redshift_arg in itertools.product(mass_args, redshift_args):
            # Compute the cluster abundance in each bin.
            last_index = 0

            print(mass_arg, redshift_arg)

            arg = np.zeros(2 + mass_arg.dim + redshift_arg.dim)
            index_map: List[int] = []
            bounds_list: List[Tuple[float, float]] = []

            if mass_arg.is_dirac_delta():
                arg[0] = mass_arg.get_logM()
            else:
                index_map.append(last_index)
                bounds_list.append(mass_arg.get_logM_bounds())
            last_index += 1

            if redshift_arg.is_dirac_delta():
                arg[1] = redshift_arg.get_z()
            else:
                index_map.append(last_index)
                bounds_list.append(redshift_arg.get_z_bounds())
            last_index += 1

            if mass_arg.dim > 0:
                index_map += list(range(last_index, last_index + mass_arg.dim))
                bounds_list += mass_arg.get_proxy_bounds()

            last_index += mass_arg.dim

            if redshift_arg.dim > 0:
                index_map += list(range(last_index, last_index + redshift_arg.dim))
                bounds_list += redshift_arg.get_proxy_bounds()
            last_index += redshift_arg.dim

            if len(index_map) == 0:
                # No proxy bins
                res.append(
                    self.mf_d2N_dz_dlnM(ccl_cosmo, arg[0], arg[1])
                    * mass_arg.p(arg[0], arg[1])
                    * redshift_arg.p(arg[0], arg[1])
                )
            else:
                res.append(
                    scipy.integrate.nquad(
                        integrand,
                        args=(index_map, arg, ccl_cosmo, mass_arg, redshift_arg),
                        ranges=bounds_list,
                        opts={"epsabs": 0.0, "epsrel": 1.0e-4},
                    )[0]
                )

        return np.array(res)

    def compute(self, ccl_cosmo: ccl.Cosmology) -> np.ndarray:
        """Computes the cluster abundance in all bins defined
        by the cluster mass and redshift proxy bins."""

        return self._compute_any(self._compute_integrand, ccl_cosmo)

    def compute_unormalized_mean_logM(self, ccl_cosmo: ccl.Cosmology) -> np.ndarray:
        """Computes the cluster abundance in all bins defined
        by the cluster mass and redshift proxy bins."""

        return self._compute_any(self._compute_integrand_mean_logM, ccl_cosmo)
