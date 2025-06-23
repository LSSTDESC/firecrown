"""The cluster kernel module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

from typing import Optional
from enum import Enum
import numpy.typing as npt
import numpy as np

from firecrown import parameters
from firecrown.updatable import Updatable

REDMAPPER_DEFAULT_AC_NC = 0.38
REDMAPPER_DEFAULT_BC_NC = 1.2634
REDMAPPER_DEFAULT_AC_MC = 13.31
REDMAPPER_DEFAULT_BC_MC = 0.2025


class KernelType(Enum):
    """The kernels that can be included in the cluster abundance integrand."""

    MASS = 1
    Z = 2
    MASS_PROXY = 3
    Z_PROXY = 4
    COMPLETENESS = 5
    PURITY = 6


class Completeness(Updatable):
    """The completeness kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the incompleteness
    of a cluster selection.
    """

    def __init__(
        self,
    ):
        super().__init__()
        # Updatable parameters
        self.ac_nc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_AC_NC
        )
        self.bc_nc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_BC_NC
        )
        self.ac_mc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_AC_MC
        )
        self.bc_mc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_BC_MC
        )

    def _mc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ac_mc = self.ac_mc
        bc_mc = self.bc_mc
        log_mc = ac_mc + bc_mc * (1.0 + z)
        mc = 10.0**log_mc
        return mc.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ac_nc = self.ac_nc
        bc_nc = self.bc_nc
        nc = ac_nc + bc_nc * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""
        mc = self._mc(z)
        mass = 10.0**log_mass
        nc = self._nc(z)
        completeness = (mass / mc) ** nc / ((mass / mc) ** nc + 1.0)
        assert isinstance(completeness, np.ndarray)
        return completeness


REDMAPPER_DEFAULT_AP_NC = 3.9193
REDMAPPER_DEFAULT_BP_NC = -0.3323
REDMAPPER_DEFAULT_AP_RC = 1.1839
REDMAPPER_DEFAULT_BP_RC = -0.4077


class Purity(Updatable):
    """The purity kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the purity
    of a cluster selection.
    """

    def __init__(self):
        super().__init__()
        self.ap_nc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_AP_NC
        )
        self.bp_nc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_BP_NC
        )
        self.ap_rc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_AP_RC
        )
        self.bp_rc = parameters.register_new_updatable_parameter(
            default_value=REDMAPPER_DEFAULT_BP_RC
        )

    def _rc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ap_rc = self.ap_rc
        bp_rc = self.bp_rc
        log_rc = ap_rc + bp_rc * (1.0 + z)
        rc = 10**log_rc
        return rc.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        bp_nc = self.bp_nc
        ap_nc = self.ap_nc
        nc = ap_nc + bp_nc * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Optional[tuple[float, float]] = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the purity contribution to the integrand."""
        if all(mass_proxy == -1.0):
            if mass_proxy_limits is None:
                raise ValueError(
                    "mass_proxy_limits must be provided when mass_proxy == -1"
                )
            mean_mass = (mass_proxy_limits[0] + mass_proxy_limits[1]) / 2
            r = np.array([np.power(10.0, mean_mass)], dtype=np.float64)
        else:
            r = np.array([np.power(10.0, mass_proxy)], dtype=np.float64)

        r_over_rc = r / self._rc(z)

        purity = (r_over_rc) ** self._nc(z) / (r_over_rc ** self._nc(z) + 1.0)
        assert isinstance(purity, np.ndarray)
        return purity


class TrueMass:
    """The true mass kernel.

    Assuming we measure the true mass, this will always be 1.
    """

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluates and returns the mass distribution contribution to the integrand.

        We have set this to 1.0 (i.e. it does not affect the mass distribution)
        """
        return np.atleast_1d(1.0)


class SpectroscopicRedshift:
    """The spec-z kernel.

    Assuming the spectroscopic redshift has no uncertainties, this is akin to
    multiplying by 1.
    """

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluates and returns the z distribution contribution to the integrand.

        We have set this to 1.0 (i.e. it does not affect the redshift distribution)
        """
        return np.atleast_1d(1.0)
