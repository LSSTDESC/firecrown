"""write me"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Optional
import numpy.typing as npt
import numpy as np
from firecrown.updatable import Updatable


class KernelType(Enum):
    """write me"""

    MASS = 1
    Z = 2
    MASS_PROXY = 3
    Z_PROXY = 4
    COMPLETENESS = 5
    PURITY = 6


class Kernel(Updatable, ABC):
    """write me"""

    def __init__(
        self,
        kernel_type: KernelType,
        is_dirac_delta: bool = False,
        has_analytic_sln: bool = False,
        integral_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.integral_bounds = integral_bounds
        self.is_dirac_delta = is_dirac_delta
        self.kernel_type = kernel_type
        self.has_analytic_sln = has_analytic_sln

    @abstractmethod
    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """The functional form of the distribution or spread of this kernel"""


class Completeness(Kernel):
    """write me"""

    def __init__(self) -> None:
        super().__init__(KernelType.COMPLETENESS)

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc * (1.0 + z)
        nc = a_nc + b_nc * (1.0 + z)
        completeness = (mass / log_mc) ** nc / ((mass / log_mc) ** nc + 1.0)
        assert isinstance(completeness, np.ndarray)
        return completeness


class Purity(Kernel):
    """write me"""

    def __init__(self) -> None:
        super().__init__(KernelType.PURITY)

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
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        if all(mass_proxy == -1.0):
            mean_mass = (mass_proxy_limits[0] + mass_proxy_limits[1]) / 2
            ln_r = np.log(10**mean_mass)
        else:
            ln_r = np.log(10**mass_proxy)

        r_over_rc = ln_r / self._ln_rc(z)

        purity = (r_over_rc) ** self._nc(z) / (r_over_rc ** self._nc(z) + 1.0)
        assert isinstance(purity, np.ndarray)
        return purity


class TrueMass(Kernel):
    """write me"""

    def __init__(self) -> None:
        super().__init__(KernelType.MASS_PROXY, True)

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        return np.atleast_1d(1.0)


class SpectroscopicRedshift(Kernel):
    """write me"""

    def __init__(self) -> None:
        super().__init__(KernelType.Z_PROXY, True)

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        return np.atleast_1d(1.0)


class DESY1PhotometricRedshift(Kernel):
    """write me"""

    def __init__(self) -> None:
        super().__init__(KernelType.Z_PROXY)
        self.sigma_0 = 0.05

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        sigma_z = self.sigma_0 * (1 + z)
        prefactor = 1 / (np.sqrt(2.0 * np.pi) * sigma_z)
        distribution = np.exp(-(1 / 2) * ((z_proxy - z) / sigma_z) ** 2.0)
        numerator = prefactor * distribution
        assert isinstance(numerator, np.ndarray)
        return numerator
