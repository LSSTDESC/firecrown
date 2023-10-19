from abc import ABC
from enum import Enum
from typing import List, Tuple

import numpy as np
import pdb


class KernelType(Enum):
    mass = 1
    z = 2
    mass_proxy = 3
    z_proxy = 4
    completeness = 5
    purity = 6


class ArgsMapping:
    def __init__(self):
        self.integral_bounds = dict()
        self.extra_args = dict()

        self.integral_bounds_idx = 0
        self.extra_args_idx = 1

    def get_integral_bounds(self, int_args, kernel_type: KernelType):
        bounds_values = int_args[self.integral_bounds_idx]
        return bounds_values[:, self.integral_bounds[kernel_type.name]]

    def get_extra_args(self, int_args, kernel_type: KernelType):
        extra_values = int_args[self.extra_args_idx]
        return extra_values[self.extra_args[kernel_type.name]]

class Parameters(Enum):
    def __init__(self):
        pass

class Kernel(ABC):
    def __init__(
        self,
        kernel_type: KernelType,
        is_dirac_delta=False,
        has_analytic_sln=False,
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        self.integral_bounds = integral_bounds
        self.is_dirac_delta = is_dirac_delta
        self.kernel_type = kernel_type
        self.has_analytic_sln = has_analytic_sln
        self.pars = Parameters()
        self.set_parameters()

    def distribution(self, args: List[float], args_map: ArgsMapping):
        raise NotImplementedError()

    def set_parameters(self):
        pass


class Completeness(Kernel):
    def __init__(self):
        super().__init__(KernelType.completeness)

    def distribution(self, args: List[float], args_map: ArgsMapping):
        mass = args_map.get_integral_bounds(args, KernelType.mass)
        z = args_map.get_integral_bounds(args, KernelType.z)

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

    def distribution(self, args: List[float], args_index_map: ArgsMapping):
        mass_proxy = args_index_map.get_integral_bounds(args, KernelType.mass_proxy)
        z = args_index_map.get_integral_bounds(args, KernelType.z)

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
    def __init__(self):
        super().__init__(KernelType.mass_proxy, True)

    def distribution(self, args: List[float], args_map: ArgsMapping):
        return 1.0


class SpectroscopicRedshift(Kernel):
    def __init__(self):
        super().__init__(KernelType.z_proxy, True)

    def distribution(self, args: List[float], args_map: ArgsMapping):
        return 1.0


class DESY1PhotometricRedshift(Kernel):
    def __init__(self):
        super().__init__(KernelType.z_proxy)
        self.sigma_0 = 0.05

    def distribution(self, args: List[float], args_map: ArgsMapping):
        z = args_map.get_integral_bounds(args, KernelType.z)
        z_proxy = args_map.get_integral_bounds(args, KernelType.z_proxy)

        sigma_z = self.sigma_0 * (1 + z)
        prefactor = 1 / (np.sqrt(2.0 * np.pi) * sigma_z)
        distribution = np.exp(-(1 / 2) * ((z_proxy - z) / sigma_z) ** 2.0)
        return prefactor * distribution
