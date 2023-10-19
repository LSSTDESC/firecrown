from typing import List, Tuple

import numpy as np

from firecrown.models.cluster.kernel import KernelType
from firecrown.models.cluster.mass_proxy.gaussian import MassRichnessGaussian


def _observed_value(
    p: Tuple[float, float, float], mass, z, pivot_mass, log1p_pivot_redshift
):
    """Return observed quantity corrected by redshift and mass."""

    ln_mass = mass * np.log(10)
    delta_ln_mass = ln_mass - pivot_mass
    delta_z = np.log1p(z) - log1p_pivot_redshift

    return p[0] + p[1] * delta_ln_mass + p[2] * delta_z


class MurataCore(MassRichnessGaussian):
    def __init__(
        self,
        pivot_mass,
        pivot_redshift,
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        super().__init__(KernelType.mass_proxy, False, True, integral_bounds)

        self.pivot_redshift = pivot_redshift
        self.pivot_mass = pivot_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

    def set_parameters(self):
        # Placeholder values
        self.pars.mu_p0 = 0.0
        self.pars.mu_p1 = 1.0
        self.pars.mu_p2 = 0.0
        self.pars.sigma_p0 = 1.0
        self.pars.sigma_p1 = 0.0
        self.pars.sigma_p2 = 1.0


    def get_proxy_mean(self, mass, z):
        """Return observed quantity corrected by redshift and mass."""
        return _observed_value(
            (self.pars.mu_p0, self.pars.mu_p1, self.pars.mu_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )

    def get_proxy_sigma(self, mass, z):
        """Return observed scatter corrected by redshift and mass."""
        return _observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            mass,
            z,
            self.pivot_mass,
            self.log1p_pivot_redshift,
        )


### used to be MassRichnessMuSigma ###
class MurataBinned(MurataCore):
    def distribution(self, args, args_map):
        return self._distribution_binned(args, args_map)


class MurataUnbinned(MurataCore):
    def distribution(self, args, args_map):
        return self._distribution_unbinned(args, args_map)
