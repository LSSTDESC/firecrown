"""Tests for the cluster deltasigma module."""

import numpy as np
import pyccl
import pytest
import clmm

from firecrown.models.cluster.deltasigma import ClusterDeltaSigma


@pytest.fixture(name="cluster_deltasigma")
def fixture_cluster_deltasigma():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterDeltaSigma((13, 17), (0, 2), hmf)
    return ca


@pytest.fixture(name="cluster_deltasigma_conc")
def fixture_cluster_deltasigma_conc():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterDeltaSigma((13, 17), (0, 2), hmf, True)
    return ca


def test_cluster_deltasigma_init(
    cluster_deltasigma: ClusterDeltaSigma, cluster_deltasigma_conc: ClusterDeltaSigma
):
    assert cluster_deltasigma is not None
    assert cluster_deltasigma_conc is not None
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


def test_deltasigma_profile_returns_value(
    cluster_deltasigma: ClusterDeltaSigma, cluster_deltasigma_conc: ClusterDeltaSigma
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)
    cluster_deltasigma_conc.update_ingredients(cosmo)
    result = cluster_deltasigma.delta_sigma(
        np.linspace(13, 17, 5, dtype=np.float64),
        np.linspace(0.1, 1, 5, dtype=np.float64),
        5.0,
    )
    cluster_deltasigma_conc.cluster_conc = 5.0
    result_conc = cluster_deltasigma_conc.delta_sigma(
        np.linspace(13, 17, 5, dtype=np.float64),
        np.linspace(0.1, 1, 5, dtype=np.float64),
        5.0,
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)

    assert isinstance(result_conc, np.ndarray)
    assert np.issubdtype(result_conc.dtype, np.float64)
    assert len(result_conc) == 5
    assert np.all(result_conc > 0)


def test_two_halo_term(cluster_deltasigma: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)
    result = cluster_deltasigma.delta_sigma(
        np.array([14.0], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        radius_center=3.0,
        two_halo_term=True,
    )
    assert result.shape == (1,)
    assert result[0] > 0


def test_miscentering(cluster_deltasigma: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)
    result = cluster_deltasigma.delta_sigma(
        np.array([14.0], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        radius_center=3.0,
        miscentering_frac=0.2,
    )
    assert result.shape == (1,)
    assert result[0] > 0

def test_miscentering_plus_two_halo_term(cluster_deltasigma: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)
    result = cluster_deltasigma.delta_sigma(
        np.array([14.0], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        radius_center=3.0,
        two_halo_term=True,
        miscentering_frac=0.2,
    )
    assert result.shape == (1,)
    assert result[0] > 0


def test_concentration_default_vs_override(cluster_deltasigma_conc: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma_conc.update_ingredients(cosmo)
    cluster_deltasigma_conc.cluster_conc = 6.0
    conc = cluster_deltasigma_conc._get_concentration(14.0, 0.5)
    assert conc == 6.0

    cluster_deltasigma_conc.conc_parameter = False
    conc = cluster_deltasigma_conc._get_concentration(14.0, 0.5)
    assert conc > 0


def test_internal_helpers(cluster_deltasigma: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)
    log_m = 14.0
    z = 0.5
    cluster_deltasigma._get_concentration(log_m, z)
    cosmo_clmm = clmm.Cosmology()
    # pylint: disable=protected-access
    cosmo_clmm._init_from_cosmo(cosmo)
    clmm_model = clmm.Modeling(
        massdef='critical',
        delta_mdef=200,
        halo_profile_model="nfw",
    )
    clmm_model.set_cosmo(cosmo_clmm)
    clmm_model.set_concentration(5.0)
    clmm_model.set_mass(10**log_m)
    result_1h = cluster_deltasigma._one_halo_contribution(clmm_model, 5.0, z)
    result_2h = cluster_deltasigma._two_halo_contribution(clmm_model, 5.0, z)
    assert isinstance(result_1h, float)
    assert isinstance(result_2h, float)
    assert result_1h > 0
    assert result_2h > 0


def test_empty_input(cluster_deltasigma: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.update_ingredients(cosmo)
    result = cluster_deltasigma.delta_sigma(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        radius_center=5.0,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)