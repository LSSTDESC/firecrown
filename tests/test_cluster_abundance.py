"""Tests for the cluster abundance module."""

import numpy as np
import pyccl
import pytest

from firecrown.models.cluster.abundance import ClusterAbundance


@pytest.fixture(name="cluster_abundance")
def fixture_cluster_abundance():
    """Test fixture that represents an assembled cluster abundance class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterAbundance((13, 17), (0, 2), hmf)
    return ca


def test_cluster_abundance_init(cluster_abundance: ClusterAbundance):
    assert cluster_abundance is not None
    assert cluster_abundance.cosmo is None
    # pylint: disable=protected-access
    assert cluster_abundance._hmf_cache == {}
    assert isinstance(
        cluster_abundance.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )
    assert cluster_abundance.min_mass == 13.0
    assert cluster_abundance.max_mass == 17.0
    assert cluster_abundance.min_z == 0.0
    assert cluster_abundance.max_z == 2.0
    assert len(cluster_abundance.kernels) == 0


def test_cluster_update_ingredients(cluster_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)
    assert cluster_abundance.cosmo is not None
    assert cluster_abundance.cosmo == cosmo
    # pylint: disable=protected-access
    assert cluster_abundance._hmf_cache == {}


def test_abundance_comoving_returns_value(cluster_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)

    result = cluster_abundance.comoving_volume(
        np.linspace(0.1, 1, 10, dtype=np.float64), 360**2
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 10
    assert np.all(result > 0)


@pytest.mark.slow
def test_abundance_massfunc_returns_value(cluster_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)

    result = cluster_abundance.mass_function(
        np.linspace(13, 17, 5, dtype=np.float64),
        np.linspace(0.1, 1, 5, dtype=np.float64),
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)
