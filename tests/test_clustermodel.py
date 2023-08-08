"""Test cluster mass function computations."""

import itertools
import numpy as np
import pytest

from firecrown.models.cluster_mass_rich_proxy import (
    ClusterMassRich,
    ClusterMassRichBinArgument,
    ClusterMassRichPointArgument,
)


@pytest.fixture(name="cluster_mass_rich")
def fixture_cluster_mass_rich() -> ClusterMassRich:
    """Initialize cluster object."""
    pivot_redshift = 0.6
    pivot_mass = 14.625862906

    cluster_mass_rich = ClusterMassRich(pivot_mass, pivot_redshift)

    # Set the parameters to the values used in the test
    # they should be such that the variance is always positive.
    cluster_mass_rich.mu_p0 = 3.00
    cluster_mass_rich.mu_p1 = 0.086
    cluster_mass_rich.mu_p2 = 0.01
    cluster_mass_rich.sigma_p0 = 3.0
    cluster_mass_rich.sigma_p1 = 0.07
    cluster_mass_rich.sigma_p2 = 0.01

    return cluster_mass_rich


def test_cluster_mass_parameters_function_z():
    for z in np.geomspace(1.0e-18, 2.0, 20):
        f_z = ClusterMassRich.cluster_mass_parameters_function(
            0.0, 0.0, (0.0, 0.0, 1.0), 0.0, z
        )
        assert f_z == pytest.approx(np.log1p(z), 1.0e-7, 0.0)


def test_cluster_mass_parameters_function_M():
    for logM in np.linspace(10.0, 16.0, 20):
        f_logM = ClusterMassRich.cluster_mass_parameters_function(
            0.0, 0.0, (0.0, 1.0, 0.0), logM, 0.0
        )

        assert f_logM == pytest.approx(logM * np.log(10.0), 1.0e-7, 0.0)


def test_cluster_mass_lnM_obs_mu_sigma(cluster_mass_rich: ClusterMassRich):
    for logM, z in itertools.product(
        np.linspace(10.0, 16.0, 20), np.geomspace(1.0e-18, 2.0, 20)
    ):
        lnM_obs_mu_direct = ClusterMassRich.cluster_mass_parameters_function(
            cluster_mass_rich.log_pivot_mass,
            cluster_mass_rich.log1p_pivot_redshift,
            (cluster_mass_rich.mu_p0, cluster_mass_rich.mu_p1, cluster_mass_rich.mu_p2),
            logM,
            z,
        )
        sigma_direct = ClusterMassRich.cluster_mass_parameters_function(
            cluster_mass_rich.log_pivot_mass,
            cluster_mass_rich.log1p_pivot_redshift,
            (
                cluster_mass_rich.sigma_p0,
                cluster_mass_rich.sigma_p1,
                cluster_mass_rich.sigma_p2,
            ),
            logM,
            z,
        )
        lnM_obs_mu, sigma = cluster_mass_rich.cluster_mass_lnM_obs_mu_sigma(logM, z)

        assert lnM_obs_mu == pytest.approx(lnM_obs_mu_direct, 1.0e-7, 0.0)
        assert sigma == pytest.approx(sigma_direct, 1.0e-7, 0.0)


def test_cluster_richness_bins_invalid_edges(cluster_mass_rich: ClusterMassRich):
    rich_bin_edges = np.array([20])

    with pytest.raises(ValueError):
        _ = cluster_mass_rich.gen_bins_by_array(rich_bin_edges)

    rich_bin_edges = np.array([20, 30, 10])
    with pytest.raises(ValueError):
        _ = cluster_mass_rich.gen_bins_by_array(rich_bin_edges)

    rich_bin_edges = np.array([20, 30, 30])
    with pytest.raises(ValueError):
        _ = cluster_mass_rich.gen_bins_by_array(rich_bin_edges)


def test_cluster_richness_bins(cluster_mass_rich: ClusterMassRich):
    rich_bin_edges = np.array([20, 40, 60, 80])

    richness_bins = cluster_mass_rich.gen_bins_by_array(rich_bin_edges)

    for Rl, Ru, richness_bin in zip(
        rich_bin_edges[:-1], rich_bin_edges[1:], richness_bins
    ):
        assert isinstance(richness_bin, ClusterMassRichBinArgument)
        assert Rl == richness_bin.logM_obs_lower
        assert Ru == richness_bin.logM_obs_upper


def test_cluster_bin_probability(cluster_mass_rich: ClusterMassRich):
    cluser_mass_rich_bin_argument = ClusterMassRichBinArgument(
        cluster_mass_rich, 13.0, 17.0, 1.0, 5.0
    )

    logM_array = np.linspace(7.0, 26.0, 2000)
    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for logM_0, logM_1 in zip(logM_array[:-1], logM_array[1:]):
            probability_0 = cluser_mass_rich_bin_argument.p(logM_0, z)
            probability_1 = cluser_mass_rich_bin_argument.p(logM_1, z)

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


def test_cluster_point_probability(cluster_mass_rich: ClusterMassRich):
    cluser_mass_rich_bin_argument = ClusterMassRichPointArgument(
        cluster_mass_rich, 13.0, 17.0, 2.5
    )

    logM_array = np.linspace(7.0, 26.0, 2000)
    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for logM_0, logM_1 in zip(logM_array[:-1], logM_array[1:]):
            probability_0 = cluser_mass_rich_bin_argument.p(logM_0, z)
            probability_1 = cluser_mass_rich_bin_argument.p(logM_1, z)

            assert probability_0 >= 0
            assert probability_1 >= 0

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
