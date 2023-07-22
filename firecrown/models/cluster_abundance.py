r"""Cluster Abundance Module
abstract class to compute cluster abundance.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import Optional, Any, Dict, List, Tuple, final

import numpy as np
import scipy.integrate
import pyccl as ccl

from ..updatable import Updatable
from ..parameters import RequiredParameters, DerivedParameterCollection
from .cluster_mass import ClusterMassArgument
from .cluster_redshift import ClusterRedshiftArgument
from numcosmo_py import Ncm

Ncm.cfg_init()

class CountsIntegralND(Ncm.Integralnd):
    """Test class for IntegralND."""
    def __init__(self, dim, fun, *args):
        super().__init__()
        self.dim = dim
        self.fun = fun
        self.args = args

    
    def do_get_dimensions(self) -> Tuple[int, int]:
        """Get number of dimensions."""
        return self.dim, 1
    
    def do_integrand(
        self,
        x_vec: Ncm.Vector,
        dim: int,
        npoints: int,
        fdim: int,
        fval_vec: Ncm.Vector
    ) -> None:
        """Integrand function."""
        x = np.array(x_vec.dup_array())
        fval = [self.fun(x, *self.args)]
        fval_vec.set_array(fval)


class ClusterAbundance(Updatable):
    r"""Cluster Abundance class"""

    def __init__(
        self,
        halo_mass_definition: ccl.halos.MassDef,
        halo_mass_function_name: str,
        halo_mass_function_args: Dict[str, Any],
        sky_area: float = 100.0,
        use_completness: bool = False,
        use_purity: bool = False,
    ):
        """Initialize the ClusterAbundance class.

        :param halo_mass_definition: Halo mass definition.
        :param halo_mass_function_name: Halo mass function name.
        :param halo_mass_function_args: Halo mass function arguments.
        :param sky_area: Sky area in square degrees, defaults to 100 sq deg.
        :param use_completness: Use completeness function, defaults to False.
        :param use_purity: Use purity function, defaults to False.

        :return: ClusterAbundance object.
        """
        super().__init__()
        self.sky_area = sky_area
        self.halo_mass_definition = halo_mass_definition
        self.halo_mass_function_name = halo_mass_function_name
        self.halo_mass_function_args = halo_mass_function_args
        self.halo_mass_function: Optional[ccl.halos.MassFunc] = None
        self.use_purity = use_purity

        if use_completness:
            self.base_mf_d2N_dz_dlnM = self.mf_d2N_dz_dlnM_completeness
        else:
            self.base_mf_d2N_dz_dlnM = self.mf_d2N_dz_dlnM

    @property
    def sky_area(self) -> float:
        """Return the sky area."""
        return self.sky_area_rad * (180.0 / np.pi) ** 2

    @sky_area.setter
    def sky_area(self, sky_area: float) -> None:
        """Set the sky area."""
        self.sky_area_rad = sky_area * (np.pi / 180.0) ** 2

    @final
    def _reset(self) -> None:
        """Implementation of the Updatable interface method `_reset`."""
        self.halo_mass_function = None

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.

        :param sacc_data: The data in the sacc format.
        """

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
            )(**self.halo_mass_function_args)
        nm = self.halo_mass_function(ccl_cosmo, mass, a)
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
        index_map, arg, ccl_cosmo, mass_arg, redshift_arg = args[-5:]
        arg[index_map] = x
        redshift_start_index = 2 + redshift_arg.dim

        logM, z = arg[0:2]
        proxy_z = arg[2:redshift_start_index]
        proxy_m = arg[redshift_start_index:]

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

    # As above but for the mean mass
    def _compute_integrand_mean_logM(self, *args) -> float:
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

    def _compute_any_from_args(
        self,
        integrand,
        ccl_cosmo: ccl.Cosmology,
        mass_arg: ClusterMassArgument,
        redshift_arg: ClusterRedshiftArgument,
    ) -> float:
        last_index = 0

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
            return (
                self.mf_d2N_dz_dlnM(ccl_cosmo, arg[0], arg[1])
                * mass_arg.p(arg[0], arg[1])
                * redshift_arg.p(arg[0], arg[1])
            )

        int_nd = CountsIntegralND(len(index_map), integrand, index_map, arg, ccl_cosmo, mass_arg, redshift_arg)
        res = Ncm.Vector.new(1)
        err = Ncm.Vector.new(1)
        int_nd.set_method(Ncm.IntegralndMethod.P)
        bound_l = []
        bound_u = []
        for item in bounds_list:
            bound_l.append(item[0])
            bound_u.append(item[1])
        int_nd.eval(Ncm.Vector.new_array(bound_l), Ncm.Vector.new_array(bound_u), res, err)
        return res.dup_array()

    def compute(
        self,
        ccl_cosmo: ccl.Cosmology,
        mass_arg: ClusterMassArgument,
        redshift_arg: ClusterRedshiftArgument,
    ) -> float:
        """Compute the integrand for the given cosmology at the given mass
        and redshift."""
        return self._compute_any_from_args(
            self._compute_integrand, ccl_cosmo, mass_arg, redshift_arg
        )

    def compute_unormalized_mean_logM(
        self,
        ccl_cosmo: ccl.Cosmology,
        mass_arg: ClusterMassArgument,
        redshift_arg: ClusterRedshiftArgument,
    ):
        """Compute the mean log(M) * integrand for the given cosmology at the
        given mass and redshift.
        """
        return self._compute_any_from_args(
            self._compute_integrand_mean_logM, ccl_cosmo, mass_arg, redshift_arg
        )
