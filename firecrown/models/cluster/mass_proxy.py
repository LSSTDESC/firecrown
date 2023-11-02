from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt

from firecrown import parameters
from scipy import special
from firecrown.models.cluster.kernel import Kernel, KernelType
from abc import abstractmethod


class MassRichnessGaussian(Kernel):
    @staticmethod
    def observed_value(
        p: Tuple[float, float, float],
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        pivot_mass: float,
        log1p_pivot_redshift: float,
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""

        ln_mass = mass * np.log(10)
        delta_ln_mass = ln_mass - pivot_mass
        delta_z = np.log1p(z) - log1p_pivot_redshift

        result = p[0] + p[1] * delta_ln_mass + p[2] * delta_z
        assert isinstance(result, np.ndarray)
        return result

    @abstractmethod
    def get_proxy_mean(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""

    @abstractmethod
    def get_proxy_sigma(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""

    def _distribution_binned(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        proxy_mean = self.get_proxy_mean(mass, z)
        proxy_sigma = self.get_proxy_sigma(mass, z)

        x_min = (proxy_mean - mass_proxy_limits[0] * np.log(10.0)) / (
            np.sqrt(2.0) * proxy_sigma
        )
        x_max = (proxy_mean - mass_proxy_limits[1] * np.log(10.0)) / (
            np.sqrt(2.0) * proxy_sigma
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
        assert isinstance(return_vals, np.ndarray)
        return return_vals

    def _distribution_unbinned(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        proxy_mean = self.get_proxy_mean(mass, z)
        proxy_sigma = self.get_proxy_sigma(mass, z)

        result = np.exp(-0.5 * (mass_proxy - proxy_mean) ** 2 / proxy_sigma**2) / (
            2 * np.pi * proxy_sigma
        )
        assert isinstance(result, np.ndarray)
        return result


class MurataBinned(MassRichnessGaussian):
    def __init__(
        self,
        pivot_mass: float,
        pivot_redshift: float,
        integral_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        super().__init__(KernelType.mass_proxy, False, True, integral_bounds)

        self.pivot_redshift = pivot_redshift
        self.pivot_mass = pivot_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        # Updatable parameters
        self.mu_p0 = parameters.register_new_updatable_parameter()
        self.mu_p1 = parameters.register_new_updatable_parameter()
        self.mu_p2 = parameters.register_new_updatable_parameter()
        self.sigma_p0 = parameters.register_new_updatable_parameter()
        self.sigma_p1 = parameters.register_new_updatable_parameter()
        self.sigma_p2 = parameters.register_new_updatable_parameter()

        # Verify this gets called last or first

    def get_proxy_mean(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.mu_p0, self.mu_p1, self.mu_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )

    def get_proxy_sigma(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        return self._distribution_binned(
            mass, z, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
        )


class MurataUnbinned(MassRichnessGaussian):
    def __init__(
        self,
        pivot_mass: float,
        pivot_redshift: float,
        integral_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        super().__init__(KernelType.mass_proxy, False, True, integral_bounds)

        self.pivot_redshift = pivot_redshift
        self.pivot_mass = pivot_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        # Updatable parameters
        self.mu_p0 = parameters.register_new_updatable_parameter()
        self.mu_p1 = parameters.register_new_updatable_parameter()
        self.mu_p2 = parameters.register_new_updatable_parameter()
        self.sigma_p0 = parameters.register_new_updatable_parameter()
        self.sigma_p1 = parameters.register_new_updatable_parameter()
        self.sigma_p2 = parameters.register_new_updatable_parameter()

        # Verify this gets called last or first

    def get_proxy_mean(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.mu_p0, self.mu_p1, self.mu_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )

    def get_proxy_sigma(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        return self._distribution_unbinned(
            mass, z, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
        )
