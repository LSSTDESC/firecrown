from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt

from firecrown import parameters
from firecrown.models.cluster.kernel import KernelType
from firecrown.models.cluster.mass_proxy.gaussian import MassRichnessGaussian


def _observed_value(
    p: Tuple[float, float, float],
    mass: npt.NDArray[np.float64],
    z: float,
    pivot_mass: float,
    log1p_pivot_redshift: float,
):
    """Return observed quantity corrected by redshift and mass."""

    ln_mass = mass * np.log(10)
    delta_ln_mass = ln_mass - pivot_mass
    delta_z = np.log1p(z) - log1p_pivot_redshift

    return p[0] + p[1] * delta_ln_mass + p[2] * delta_z


class MurataCore(MassRichnessGaussian):
    def __init__(
        self,
        pivot_mass: float,
        pivot_redshift: float,
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        super().__init__(KernelType.mass_proxy, False, True, integral_bounds)

        self.pivot_redshift = pivot_redshift
        self.pivot_mass = pivot_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        # Updatable parameters
        self.mu_p0 = parameters.create()
        self.mu_p1 = parameters.create()
        self.mu_p2 = parameters.create()
        self.sigma_p0 = parameters.create()
        self.sigma_p1 = parameters.create()
        self.sigma_p2 = parameters.create()

        # Verify this gets called last or first

    def get_proxy_mean(
        self,
        mass: Union[float, npt.NDArray[np.float64]],
        z: Union[float, npt.NDArray[np.float64]],
    ):
        """Return observed quantity corrected by redshift and mass."""
        return _observed_value(
            (self.mu_p0, self.mu_p1, self.mu_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )

    def get_proxy_sigma(
        self,
        mass: Union[float, npt.NDArray[np.float64]],
        z: Union[float, npt.NDArray[np.float64]],
    ):
        """Return observed scatter corrected by redshift and mass."""
        return _observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )


class MurataBinned(MurataCore):
    def distribution(self, args, args_map):
        return self._distribution_binned(args, args_map)


class MurataUnbinned(MurataCore):
    def distribution(self, args, args_map):
        return self._distribution_unbinned(args, args_map)
