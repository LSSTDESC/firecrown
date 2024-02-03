"""The cluster kernel module

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand."""

from enum import Enum
from typing import Tuple
import numpy.typing as npt
import numpy as np


class KernelType(Enum):
    """The kernels that can be included in the cluster abundance integrand"""

    MASS = 1
    Z = 2
    MASS_PROXY = 3
    Z_PROXY = 4
    COMPLETENESS = 5
    PURITY = 6


# pylint: disable=too-few-public-methods
class Completeness:
    """The completeness kernel for the numcosmo simulated survey

    This kernel will affect the integrand by accounting for the incompleteness
    of a cluster selection."""

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc * (1.0 + z)
        nc = a_nc + b_nc * (1.0 + z)
        completeness = (mass / log_mc) ** nc / ((mass / log_mc) ** nc + 1.0)
        assert isinstance(completeness, np.ndarray)
        return completeness


# pylint: disable=too-few-public-methods
class Purity:
    """The purity kernel for the numcosmo simulated survey

    This kernel will affect the integrand by accounting for the purity
    of a cluster selection."""

    def _ln_rc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a_rc = 2.2183
        b_rc = -0.6592
        ln_rc = a_rc + b_rc * (1.0 + z)
        return ln_rc

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        b_nc = np.log(10) * 0.3527
        a_nc = np.log(10) * 0.8612
        nc = a_nc + b_nc * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the purity contribution to the integrand."""
        if all(mass_proxy == -1.0):
            mean_mass = (mass_proxy_limits[0] + mass_proxy_limits[1]) / 2
            ln_r = np.log(10**mean_mass)
        else:
            ln_r = np.log(10**mass_proxy)

        r_over_rc = ln_r / self._ln_rc(z)

        purity = (r_over_rc) ** self._nc(z) / (r_over_rc ** self._nc(z) + 1.0)
        assert isinstance(purity, np.ndarray)
        return purity


# pylint: disable=too-few-public-methods
class TrueMass:
    """The true mass kernel.

    Assuming we measure the true mass, this will always be 1."""

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluates and returns the mass distribution contribution to the integrand.
        We have set this to 1.0 (i.e. it does not affect the mass distribution)"""
        return np.atleast_1d(1.0)


# pylint: disable=too-few-public-methods
class SpectroscopicRedshift:
    """The spec-z kernel.

    Assuming the spectroscopic redshift has no uncertainties, this is akin to
    multiplying by 1."""

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluates and returns the z distribution contribution to the integrand.
        We have set this to 1.0 (i.e. it does not affect the redshift distribution)"""
        return np.atleast_1d(1.0)
