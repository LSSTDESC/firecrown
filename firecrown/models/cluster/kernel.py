from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy.typing as npt
import numpy as np
from firecrown.updatable import Updatable


class KernelType(Enum):
    mass = 1
    z = 2
    mass_proxy = 3
    z_proxy = 4
    completeness = 5
    purity = 6


class ArgReader(ABC):
    def __init__(self) -> None:
        self.integral_bounds: Dict[str, int] = dict()
        self.extra_args: Dict[str, int] = dict()

    @abstractmethod
    def get_independent_val(
        self,
        integral_args: Tuple[Any, ...],
        kernel_type: KernelType,
    ) -> Union[float, npt.NDArray[np.float64]]:
        """Returns the current differential value for KernelType"""

    @abstractmethod
    def get_extra_args(
        self,
        integral_args: Tuple[Any, ...],
        kernel_type: KernelType,
    ) -> Union[float, Tuple[float, float]]:
        """Returns the extra arguments passed into the integral for KernelType"""


class Kernel(Updatable, ABC):
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
    def __init__(self) -> None:
        super().__init__(KernelType.completeness)

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
        return completeness


class Purity(Kernel):
    def __init__(self) -> None:
        super().__init__(KernelType.purity)

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        a_nc = np.log(10) * 0.8612
        b_nc = np.log(10) * 0.3527
        a_rc = 2.2183
        b_rc = -0.6592

        ln_r = np.log(10**mass_proxy)
        ln_rc = a_rc + b_rc * (1.0 + z)
        r_over_rc = ln_r / ln_rc

        nc = a_nc + b_nc * (1.0 + z)

        purity = (r_over_rc) ** nc / (r_over_rc**nc + 1.0)
        return purity


class TrueMass(Kernel):
    def __init__(self) -> None:
        super().__init__(KernelType.mass_proxy, True)

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
    def __init__(self) -> None:
        super().__init__(KernelType.z_proxy, True)

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
    def __init__(self) -> None:
        super().__init__(KernelType.z_proxy)
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
        return prefactor * distribution
