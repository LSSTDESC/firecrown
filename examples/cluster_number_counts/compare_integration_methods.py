"""Test integral methods for cluster abundance."""

import time
import itertools

import pyccl
import numpy as np
from firecrown.models.cluster.integrator.numcosmo_integrator import (
    NumCosmoIntegralMethod,
)
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.recipes.murata_binned_spec_z import (
    MurataBinnedSpecZRecipe,
)
from firecrown.models.cluster.binning import TupleBin


def get_cosmology() -> pyccl.Cosmology:
    """Creates and returns a CCL cosmology object."""
    Omega_c = 0.262
    Omega_b = 0.049
    Omega_k = 0.0
    H0 = 67.66
    h = H0 / 100.0
    Tcmb0 = 2.7255
    sigma8 = 0.8277
    n_s = 0.96
    Neff = 3.046
    w0 = -1.0
    wa = 0.0

    cosmo_ccl = pyccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        Neff=Neff,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
        Omega_k=Omega_k,
        w0=w0,
        wa=wa,
        T_CMB=Tcmb0,
        m_nu=[0.00, 0.0, 0.0],
    )

    return cosmo_ccl


def compare_integration() -> None:
    """Compare integration methods."""
    hmf = pyccl.halos.MassFuncTinker08()
    abundance = ClusterAbundance(13, 15, 0, 4, hmf)
    cluster_recipe = MurataBinnedSpecZRecipe()

    cluster_recipe.mass_distribution.mu_p0 = 3.0
    cluster_recipe.mass_distribution.mu_p1 = 0.86
    cluster_recipe.mass_distribution.mu_p2 = 0.0
    cluster_recipe.mass_distribution.sigma_p0 = 3.0
    cluster_recipe.mass_distribution.sigma_p1 = 0.7
    cluster_recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = get_cosmology()
    abundance.update_ingredients(cosmo)

    sky_area = 360**2

    for method in NumCosmoIntegralMethod:
        counts_list = []
        t_start = time.time()

        cluster_recipe.integrator.method = method

        # nc_integrator.set_integration_bounds(abundance, 496, (0, 4), (13, 15))

        z_bins = np.linspace(0.0, 1.0, 4)
        mass_proxy_bins = np.linspace(1.0, 2.5, 5)

        for z_idx, mass_proxy_idx in itertools.product(range(3), range(4)):
            mass_limits = (
                mass_proxy_bins[mass_proxy_idx],
                mass_proxy_bins[mass_proxy_idx + 1],
            )
            z_limits = (z_bins[z_idx], z_bins[z_idx + 1])

            tuple_bin = TupleBin([mass_limits, z_limits])

            counts = cluster_recipe.evaluate_theory_prediction(
                abundance, tuple_bin, sky_area
            )
            counts_list.append(counts)

        t_stop = time.time()
        delta_t = t_stop - t_start

        print(
            f"The time for NumCosmo integration method {method} is {delta_t}\n\n"
            f"The counts value is {counts_list}\n\n"
        )


if __name__ == "__main__":
    compare_integration()
