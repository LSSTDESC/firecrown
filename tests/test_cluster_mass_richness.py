"""Tests for the cluster mass richness module."""

import pytest
import numpy as np
from scipy.integrate import quad
from firecrown.models.cluster.mass_proxy import (
    MurataBinned,
    MurataUnbinned,
    MassRichnessGaussian,
)


PIVOT_Z = 0.6
PIVOT_MASS = 14.625862906


@pytest.fixture(name="murata_binned_relation")
def fixture_murata_binned() -> MurataBinned:
    """Initialize cluster object."""

    mr = MurataBinned(PIVOT_MASS, PIVOT_Z)

    # Set the parameters to the values used in the test
    # they should be such that the variance is always positive.
    mr.mu_p0 = 3.00
    mr.mu_p1 = 0.086
    mr.mu_p2 = 0.01
    mr.sigma_p0 = 3.0
    mr.sigma_p1 = 0.07
    mr.sigma_p2 = 0.01

    return mr


@pytest.fixture(name="murata_unbinned_relation")
def fixture_murata_unbinned() -> MurataUnbinned:
    """Initialize cluster object."""

    mr = MurataUnbinned(PIVOT_MASS, PIVOT_Z)

    # Set the parameters to the values used in the test
    # they should be such that the variance is always positive.
    mr.mu_p0 = 3.00
    mr.mu_p1 = 0.086
    mr.mu_p2 = 0.01
    mr.sigma_p0 = 3.0
    mr.sigma_p1 = 0.07
    mr.sigma_p2 = 0.01

    return mr


def test_create_musigma_kernel():
    mb = MurataBinned(1, 1)
    assert mb.pivot_mass == 1 * np.log(10)
    assert mb.pivot_redshift == 1
    assert mb.log1p_pivot_redshift == np.log1p(1)

    assert mb.mu_p0 is None
    assert mb.mu_p1 is None
    assert mb.mu_p2 is None
    assert mb.sigma_p0 is None
    assert mb.sigma_p1 is None
    assert mb.sigma_p2 is None


def test_cluster_observed_z():
    for z in np.geomspace(1.0e-18, 2.0, 20):
        z = np.atleast_1d(z)
        mass = np.atleast_1d(0)
        f_z = MassRichnessGaussian.observed_value((0.0, 0.0, 1.0), mass, z, 0, 0)
        assert f_z == pytest.approx(np.log1p(z), 1.0e-7, 0.0)


def test_cluster_observed_mass():
    for mass in np.linspace(10.0, 16.0, 20):
        z = np.atleast_1d(0)
        mass = np.atleast_1d(mass)
        f_logM = MassRichnessGaussian.observed_value((0.0, 1.0, 0.0), mass, z, 0, 0)

        assert f_logM == pytest.approx(mass * np.log(10.0), 1.0e-7, 0.0)


def test_cluster_murata_binned_distribution(murata_binned_relation: MurataBinned):
    mass_array = np.linspace(7.0, 26.0, 20, dtype=np.float64)
    mass_proxy_limits = (1.0, 5.0)

    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for mass1, mass2 in zip(mass_array[:-1], mass_array[1:]):
            mass1_a = np.atleast_1d(mass1)
            mass2_a = np.atleast_1d(mass2)
            z = np.atleast_1d(z)

            probability_0 = murata_binned_relation.distribution(
                mass1_a, z, mass_proxy_limits
            )
            probability_1 = murata_binned_relation.distribution(
                mass2_a, z, mass_proxy_limits
            )

            assert probability_0 >= 0
            assert probability_1 >= 0

            # Probability should be initially monotonically increasing
            # and then monotonically decreasing. It should flip only once.

            # Test for the flip
            if (not flip) and (probability_1 < probability_0):
                flip = True

            # Test for the second flip
            if flip and (probability_1 > probability_0):
                raise ValueError("Probability flipped twice")

            if flip:
                assert probability_1 <= probability_0
            else:
                assert probability_1 >= probability_0


def test_cluster_murata_binned_mean(murata_binned_relation: MurataBinned):
    for mass in np.linspace(7.0, 26.0, 20):
        for z in np.geomspace(1.0e-18, 2.0, 20):
            mass = np.atleast_1d(mass)
            z = np.atleast_1d(z)
            test = murata_binned_relation.get_proxy_mean(mass, z)

            true = MassRichnessGaussian.observed_value(
                (3.00, 0.086, 0.01),
                mass,
                z,
                PIVOT_MASS * np.log(10.0),
                np.log1p(PIVOT_Z),
            )

            assert test == pytest.approx(true, rel=1e-7, abs=0.0)


def test_cluster_murata_binned_variance(murata_binned_relation: MurataBinned):
    for mass in np.linspace(7.0, 26.0, 20):
        for z in np.geomspace(1.0e-18, 2.0, 20):
            mass = np.atleast_1d(mass)
            z = np.atleast_1d(z)
            test = murata_binned_relation.get_proxy_sigma(mass, z)

            true = MassRichnessGaussian.observed_value(
                (3.00, 0.07, 0.01),
                mass,
                z,
                PIVOT_MASS * np.log(10.0),
                np.log1p(PIVOT_Z),
            )

            assert test == pytest.approx(true, rel=1e-7, abs=0.0)


def test_cluster_murata_unbinned_distribution(murata_unbinned_relation: MurataUnbinned):
    mass_array = np.linspace(7.0, 26.0, 20, dtype=np.float64)

    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for mass1, mass2 in zip(mass_array[:-1], mass_array[1:]):
            mass1_a = np.atleast_1d(mass1)
            mass2_a = np.atleast_1d(mass2)
            z = np.atleast_1d(z)
            mass_proxy = np.atleast_1d(1)

            probability_0 = murata_unbinned_relation.distribution(
                mass1_a, z, mass_proxy
            )
            probability_1 = murata_unbinned_relation.distribution(
                mass2_a, z, mass_proxy
            )

            # Probability density should be initially monotonically increasing
            # and then monotonically decreasing. It should flip only once.

            # Test for the flip
            if (not flip) and (probability_1 < probability_0):
                flip = True

            # Test for the second flip
            if flip and (probability_1 > probability_0):
                raise ValueError("Probability flipped twice")

            if flip:
                assert probability_1 <= probability_0
            else:
                assert probability_1 >= probability_0


@pytest.mark.precision_sensitive
def test_cluster_murata_unbinned_distribution_is_normalized(
    murata_unbinned_relation: MurataUnbinned,
):
    for mass_i, z_i in zip(np.linspace(10.0, 16.0, 20), np.geomspace(1.0e-18, 2.0, 20)):
        mass = np.atleast_1d(mass_i)
        z = np.atleast_1d(z_i)

        mean = murata_unbinned_relation.get_proxy_mean(mass, z)[0]
        sigma = murata_unbinned_relation.get_proxy_sigma(mass, z)[0]
        mass_proxy_limits = np.array([mean - 5 * sigma, mean + 5 * sigma])
        print(mass_proxy_limits, mean, sigma)

        def integrand(ln_mass_proxy) -> float:
            """Evaluate the unbinned distribution at fixed mass and redshift."""
            log10_mass_proxy = ln_mass_proxy / np.log(10.0)
            # pylint: disable=cell-var-from-loop
            return murata_unbinned_relation.distribution(mass, z, log10_mass_proxy)[0]

        result, _ = quad(
            integrand,
            mass_proxy_limits[0],
            mass_proxy_limits[1],
            epsabs=1e-12,
            epsrel=1e-12,
        )

        assert result == pytest.approx(1.0, rel=1.0e-6, abs=0.0)
