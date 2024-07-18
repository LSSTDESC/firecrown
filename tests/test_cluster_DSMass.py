"""Tests for the cluster abundance module."""

import numpy as np
import pyccl
import pytest

from firecrown.models.cluster.DS_from_mass import DS_from_mass


@pytest.fixture(name="ds_from_mass")
def fixture_ds_from_mass():
    """Test fixture that represents an assembled cluster abundance class."""
    ds = DS_from_mass()
    return ds

def test_ds_from_mass_init(ds: DS_from_mass):
    assert ds is not None
    assert ds.cosmo is None
        ds.moo, clmm.modeling
    )


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

    result = cluster_abundance.comoving_volume(np.linspace(0.1, 1, 10), 360**2)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 10
    assert np.all(result > 0)


@pytest.mark.slow
def test_abundance_massfunc_returns_value(cluster_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)

    result = cluster_abundance.mass_function(
        np.linspace(13, 17, 5), np.linspace(0.1, 1, 5)
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)
