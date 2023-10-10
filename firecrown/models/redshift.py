import numpy as np
from firecrown.models.kernel import Kernel, KernelType
from typing import List, Tuple, Dict


class Redshift(Kernel):
    def __init__(self, integral_bounds: List[Tuple[float, float]] = None):
        super().__init__(KernelType.z, integral_bounds)

    def distribution(self, args: List[float], index_lkp: Dict[str, int]):
        return 1.0


class SpectroscopicRedshift(Kernel):
    def __init__(self, integral_bounds: List[Tuple[float, float]] = None):
        super().__init__(KernelType.z_proxy, integral_bounds)

    def distribution(self, args: List[float], index_lkp: Dict[str, int]):
        return 1.0


class DESY1PhotometricRedshift(Kernel):
    def __init__(self, integral_bounds: List[Tuple[float, float]] = None):
        super().__init__(KernelType.z_proxy, integral_bounds)
        self.sigma_0 = 0.05

    def distribution(self, args: List[float], index_lkp: Dict[str, int]):
        z_proxy = args[index_lkp["z_proxy"]]
        z = args[index_lkp["z"]]

        sigma_z = self.sigma_0 * (1 + z)
        prefactor = 1 / (np.sqrt(2.0 * np.pi) * sigma_z)
        distribution = np.exp(-(1 / 2) * ((z_proxy - z) / sigma_z) ** 2.0)
        return prefactor * distribution
