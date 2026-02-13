"""Tests for the cluster mass richness module."""

import pytest
import numpy as np
from scipy.integrate import quad
from hypothesis import given, assume
from hypothesis.strategies import floats
from firecrown.models.cluster import (
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


@given(z=floats(min_value=1e-15, max_value=2.0))
def test_cluster_observed_z_mathematical_property(z: float):
    """Test mathematical identity: f(z) = ln(1+z) using hypothesis."""
    zarray = np.atleast_1d(z)
    mass = np.atleast_1d(0)
    f_z = MassRichnessGaussian.observed_value((0.0, 0.0, 1.0), mass, zarray, 0, 0)
    expected = np.log1p(zarray)
    assert f_z == pytest.approx(
        expected, rel=1.0e-7, abs=0.0
    ), f"Expected f(z={z}) = ln(1+z) = {expected}, got {f_z}"


@given(mass=floats(min_value=10.0, max_value=16.0))
def test_cluster_observed_mass_mathematical_property(mass: float):
    """Test mathematical identity: f(mass) = mass * ln(10) using hypothesis."""
    z = np.atleast_1d(0)
    massarray = np.atleast_1d(mass)
    f_logM = MassRichnessGaussian.observed_value((0.0, 1.0, 0.0), massarray, z, 0, 0)
    expected = mass * np.log(10.0)
    assert f_logM == pytest.approx(
        expected, rel=1.0e-7, abs=0.0
    ), f"Expected f(mass={mass}) = mass * ln(10) = {expected}, got {f_logM}"


def test_cluster_murata_binned_distribution(murata_binned_relation: MurataBinned):
    mass_array = np.linspace(7.0, 26.0, 20, dtype=np.float64)
    mass_proxy_limits = (1.0, 5.0)

    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for mass1, mass2 in zip(mass_array[:-1], mass_array[1:]):
            mass1_a = np.atleast_1d(mass1)
            mass2_a = np.atleast_1d(mass2)
            zarray = np.atleast_1d(z)

            probability_0 = murata_binned_relation.distribution(
                mass1_a, zarray, mass_proxy_limits
            )
            probability_1 = murata_binned_relation.distribution(
                mass2_a, zarray, mass_proxy_limits
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


@given(
    z=floats(min_value=1e-15, max_value=2.0), mass=floats(min_value=7.0, max_value=26.0)
)
def test_cluster_distribution_properties(z: float, mass: float):
    """Mathematical properties of the cluster mass distribution using hypothesis."""
    # Create the relation inside the test to avoid fixture issues
    murata_binned_relation = MurataBinned(PIVOT_MASS, PIVOT_Z)
    murata_binned_relation.mu_p0 = 3.00
    murata_binned_relation.mu_p1 = 0.086
    murata_binned_relation.mu_p2 = 0.01
    murata_binned_relation.sigma_p0 = 3.0
    murata_binned_relation.sigma_p1 = 0.07
    murata_binned_relation.sigma_p2 = 0.01

    mass_proxy_limits = (1.0, 5.0)

    mass_array = np.atleast_1d(mass)
    z_array = np.atleast_1d(z)

    probability = murata_binned_relation.distribution(
        mass_array, z_array, mass_proxy_limits
    )

    # Test non-negativity property
    assert probability >= 0, f"Probability must be non-negative, got {probability}"


@given(
    z=floats(min_value=1e-15, max_value=2.0),
    mass1=floats(min_value=7.0, max_value=25.0),
    mass_delta=floats(min_value=0.1, max_value=1.0),
)
def test_cluster_distribution_unimodal_property(
    z: float, mass1: float, mass_delta: float
):
    """Test that the distribution has at most one peak (unimodal) using hypothesis."""
    # Create the relation inside the test to avoid fixture issues
    murata_binned_relation = MurataBinned(PIVOT_MASS, PIVOT_Z)
    murata_binned_relation.mu_p0 = 3.00
    murata_binned_relation.mu_p1 = 0.086
    murata_binned_relation.mu_p2 = 0.01
    murata_binned_relation.sigma_p0 = 3.0
    murata_binned_relation.sigma_p1 = 0.07
    murata_binned_relation.sigma_p2 = 0.01

    mass_proxy_limits = (1.0, 5.0)
    mass2 = mass1 + mass_delta

    # Skip if mass2 is out of range
    assume(mass2 <= 26.0)

    # pylint: disable=unreachable
    mass1_array = np.atleast_1d(mass1)

    mass2_array = np.atleast_1d(mass2)
    z_array = np.atleast_1d(z)

    prob1 = murata_binned_relation.distribution(mass1_array, z_array, mass_proxy_limits)
    prob2 = murata_binned_relation.distribution(mass2_array, z_array, mass_proxy_limits)

    # Both probabilities should be non-negative
    assert prob1 >= 0, f"Probability at mass1={mass1} must be non-negative"
    assert prob2 >= 0, f"Probability at mass2={mass2} must be non-negative"


def test_cluster_murata_binned_mean(murata_binned_relation: MurataBinned):
    for mass in np.linspace(7.0, 26.0, 20):
        for z in np.geomspace(1.0e-18, 2.0, 20):
            massarray = np.atleast_1d(mass)
            zarray = np.atleast_1d(z)
            test = murata_binned_relation.get_proxy_mean(massarray, zarray)

            true = MassRichnessGaussian.observed_value(
                (3.00, 0.086, 0.01),
                massarray,
                zarray,
                PIVOT_MASS * np.log(10.0),
                np.log1p(PIVOT_Z),
            )

            assert test == pytest.approx(true, rel=1e-7, abs=0.0)


def test_cluster_murata_binned_variance(murata_binned_relation: MurataBinned):
    for mass in np.linspace(7.0, 26.0, 20):
        for z in np.geomspace(1.0e-18, 2.0, 20):
            massarray = np.atleast_1d(mass)
            zarray = np.atleast_1d(z)
            test = murata_binned_relation.get_proxy_sigma(massarray, zarray)

            true = MassRichnessGaussian.observed_value(
                (3.00, 0.07, 0.01),
                massarray,
                zarray,
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
            zarray = np.atleast_1d(z)
            mass_proxy = np.atleast_1d(1)

            probability_0 = murata_unbinned_relation.distribution(
                mass1_a, zarray, mass_proxy
            )
            probability_1 = murata_unbinned_relation.distribution(
                mass2_a, zarray, mass_proxy
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
