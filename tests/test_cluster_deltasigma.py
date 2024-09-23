"""Tests for the cluster deltasigma module."""

import numpy as np
import pyccl
import pytest

from firecrown.models.cluster.deltasigma import ClusterDeltaSigma


@pytest.fixture(name="cluster_deltasigma")
def fixture_cluster_deltasigma():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterDeltaSigma(13, 17, 0, 2, hmf)
    return ca


def test_cluster_deltasigma_init(cluster_deltasigma: ClusterDeltaSigma):
    assert cluster_deltasigma is not None
    assert cluster_deltasigma.cosmo is None
    # pylint: disable=protected-access
    assert cluster_deltasigma._hmf_cache == {}
    assert isinstance(
        cluster_deltasigma.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )
    assert cluster_deltasigma.min_mass == 13.0
    assert cluster_deltasigma.max_mass == 17.0
    assert cluster_deltasigma.min_z == 0.0
    assert cluster_deltasigma.max_z == 2.0
    assert len(cluster_deltasigma.kernels) == 0


def test_cluster_update_ingredients(cluster_deltasigma: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)
    assert cluster_deltasigma.cosmo is not None
    assert cluster_deltasigma.cosmo == cosmo
    # pylint: disable=protected-access
    assert cluster_deltasigma._hmf_cache == {}


@pytest.mark.slow
def test_deltasigma_profile_returns_value(cluster_deltasigma: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)

    result = cluster_deltasigma.delta_sigma(
        np.linspace(13, 17, 5), np.linspace(0.1, 1, 5), np.linspace(1, 5, 5)
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)
