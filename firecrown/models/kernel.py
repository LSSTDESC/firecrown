from typing import List, Tuple, Dict
import numpy as np
from firecrown.updatable import Updatable
from abc import ABC
from enum import Enum


class KernelType(Enum):
    mass = 1
    z = 2
    mass_proxy = 3
    z_proxy = 4
    completeness = 5
    purity = 6


class Kernel(Updatable, ABC):
    def __init__(
        self,
        kernel_type: KernelType,
        is_dirac_delta=False,
        has_analytic_sln=False,
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.integral_bounds = integral_bounds
        self.is_dirac_delta = is_dirac_delta
        self.kernel_type = kernel_type
        self.has_analytic_sln = has_analytic_sln

    def distribution(self, args: List[float], args_map: Dict[str, int]):
        raise NotImplementedError()

    def analytic_solution(self, args: List[float], args_map: Dict[str, int]):
        raise NotImplementedError()


class Completeness(Kernel):
    def __init__(self):
        super().__init__(KernelType.completeness)

    def distribution(self, args: List[float], args_map: Dict[str, int]):
        mass = args[args_map[KernelType.mass.name]]
        z = args[args_map[KernelType.z.name]]
        # TODO improve parameter names
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc * (1.0 + z)
        nc = a_nc + b_nc * (1.0 + z)
        completeness = (mass / log_mc) ** nc / ((mass / log_mc) ** nc + 1.0)
        return completeness


class Purity(Kernel):
    def __init__(self):
        super().__init__(KernelType.purity)

    def distribution(self, args: List[float], index_lkp: Dict[str, int]):
        mass_proxy = args[index_lkp[KernelType.mass_proxy.name]]
        z = args[index_lkp[KernelType.z.name]]

        ln_r = np.log(10**mass_proxy)
        a_nc = np.log(10) * 0.8612
        b_nc = np.log(10) * 0.3527
        a_rc = 2.2183
        b_rc = -0.6592
        nc = a_nc + b_nc * (1.0 + z)
        ln_rc = a_rc + b_rc * (1.0 + z)
        purity = (ln_r / ln_rc) ** nc / ((ln_r / ln_rc) ** nc + 1.0)
        return purity


class Miscentering(Kernel):
    pass
