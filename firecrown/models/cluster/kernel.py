from abc import ABC
from enum import Enum
from typing import List, Tuple

import numpy as np
from scipy import special

from firecrown import parameters
from firecrown.updatable import Updatable
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

    def distribution(self, args: List[float], args_map: ArgsMapping):
        raise NotImplementedError()


class Completeness(Kernel):
    def __init__(self):
        super().__init__(KernelType.completeness)

    def distribution(self, args: List[float], args_map: ArgsMapping):
        mass = args_map.get_integral_bounds(args, KernelType.mass)
        z = args_map.get_integral_bounds(args, KernelType.z)

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

    def distribution(self, args: List[float], index_lkp: ArgsMapping):
        mass_proxy = args[index_lkp.integral_bounds[KernelType.mass_proxy.name]]
        z = args[index_lkp.integral_bounds[KernelType.z.name]]

        ln_r = np.log(10**mass_proxy)
        a_nc = np.log(10) * 0.8612
        b_nc = np.log(10) * 0.3527
        a_rc = 2.2183
        b_rc = -0.6592
        nc = a_nc + b_nc * (1.0 + z)
        ln_rc = a_rc + b_rc * (1.0 + z)
        purity = (ln_r / ln_rc) ** nc / ((ln_r / ln_rc) ** nc + 1.0)
        return purity


class MassRichnessMuSigma(Kernel):
    def __init__(
        self,
        pivot_mass,
        pivot_redshift,
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        super().__init__(KernelType.mass_proxy, False, True, integral_bounds)
        self.pivot_mass = pivot_mass
        self.pivot_redshift = pivot_redshift
        self.pivot_mass = self.pivot_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        # Updatable parameters
        self.mu_p0 = parameters.create()
        self.mu_p1 = parameters.create()
        self.mu_p2 = parameters.create()
        self.sigma_p0 = parameters.create()
        self.sigma_p1 = parameters.create()
        self.sigma_p2 = parameters.create()

        # Verify this gets called last or first

    def observed_value(self, p: Tuple[float, float, float], mass, z):
        """Return observed quantity corrected by redshift and mass."""

        ln_mass = mass * np.log(10)
        delta_ln_mass = ln_mass - self.pivot_mass
        delta_z = np.log1p(z) - self.log1p_pivot_redshift

        return p[0] + p[1] * delta_ln_mass + p[2] * delta_z

    def distribution(self, args: List[float], args_map: ArgsMapping):
        mass = args_map.get_integral_bounds(args, KernelType.mass)
        z = args_map.get_integral_bounds(args, KernelType.z)
        mass_limits = args_map.get_extra_args(args, self.kernel_type)
        observed_mean_mass = self.observed_value(
            (self.mu_p0, self.mu_p1, self.mu_p2),
            mass,
            z,
        )
        observed_mass_sigma = self.observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            mass,
            z,
        )

        x_min = (observed_mean_mass - mass_limits[0] * np.log(10.0)) / (
            np.sqrt(2.0) * observed_mass_sigma
        )
        x_max = (observed_mean_mass - mass_limits[1] * np.log(10.0)) / (
            np.sqrt(2.0) * observed_mass_sigma
        )

        return_vals = np.empty_like(x_min)
        mask1 = (x_max > 3.0) | (x_min < -3.0)
        mask2 = ~mask1

        return_vals[mask1] = (
            -(special.erfc(x_min[mask1]) - special.erfc(x_max[mask1])) / 2.0
        )
        return_vals[mask2] = (
            special.erf(x_min[mask2]) - special.erf(x_max[mask2])
        ) / 2.0

        return return_vals


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
