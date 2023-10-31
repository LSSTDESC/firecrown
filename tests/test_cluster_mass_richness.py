import pytest
import numpy as np
from firecrown.models.cluster.mass_proxy import (
    MurataBinned,
    MurataUnbinned,
    MassRichnessGaussian,
)
from firecrown.models.cluster.kernel import (
    KernelType,
    Kernel,
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
    assert isinstance(mb, Kernel)
    assert mb.kernel_type == KernelType.mass_proxy
    assert mb.is_dirac_delta is False
    assert mb.integral_bounds is None
    assert mb.has_analytic_sln is True
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
        f_z = MassRichnessGaussian.observed_value((0.0, 0.0, 1.0), 0.0, z, 0, 0)
        assert f_z == pytest.approx(np.log1p(z), 1.0e-7, 0.0)


def test_cluster_observed_mass():
    for logM in np.linspace(10.0, 16.0, 20):
        f_logM = MassRichnessGaussian.observed_value((0.0, 1.0, 0.0), logM, 0.0, 0, 0)

        assert f_logM == pytest.approx(logM * np.log(10.0), 1.0e-7, 0.0)


def test_cluster_murata_binned_distribution(murata_binned_relation: MurataBinned):
    mass_array = np.linspace(7.0, 26.0, 20)
    mass_proxy_limits = (1.0, 5.0)
    z_proxy_limits = (0.0, 1.0)

    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for mass1, mass2 in zip(mass_array[:-1], mass_array[1:]):
            mass1 = np.atleast_1d(mass1)
            mass2 = np.atleast_1d(mass2)
            z = np.atleast_1d(z)
            z_proxy = np.atleast_1d(0)
            mass_proxy = np.atleast_1d(1)

            probability_0 = murata_binned_relation.distribution(
                mass1, z, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
            )
            probability_1 = murata_binned_relation.distribution(
                mass2, z, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
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
    mass_array = np.linspace(7.0, 26.0, 20)
    mass_proxy_limits = (1.0, 5.0)
    z_proxy_limits = (0.0, 1.0)

    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for mass1, mass2 in zip(mass_array[:-1], mass_array[1:]):
            mass1 = np.atleast_1d(mass1)
            mass2 = np.atleast_1d(mass2)
            z = np.atleast_1d(z)
            z_proxy = np.atleast_1d(0)
            mass_proxy = np.atleast_1d(1)

            probability_0 = murata_unbinned_relation.distribution(
                mass1, z, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
            )
            probability_1 = murata_unbinned_relation.distribution(
                mass2, z, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
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
