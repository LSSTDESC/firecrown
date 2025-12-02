"""Unit tests for the cluster recipes module."""

from unittest.mock import Mock

import numpy as np
import pyccl
import pytest

from firecrown.models.cluster import (
    ClusterAbundance,
    ClusterDeltaSigma,
    ClusterProperty,
    ClusterRecipe,
    MurataBinned,
    MurataBinnedSpecZRecipe,
    MurataBinnedSpecZDeltaSigmaRecipe,
    NDimensionalBin,
    NumCosmoIntegrator,
    SpectroscopicRedshift,
)


def test_murata_binned_spec_z_init():
    """Test initialization of MurataBinnedSpecZRecipe."""
    recipe = MurataBinnedSpecZRecipe()

    assert recipe is not None
    assert isinstance(recipe, ClusterRecipe)
    assert recipe.integrator is not None
    assert isinstance(recipe.integrator, NumCosmoIntegrator)
    assert recipe.redshift_distribution is not None
    assert isinstance(recipe.redshift_distribution, SpectroscopicRedshift)
    assert recipe.mass_distribution is not None
    assert isinstance(recipe.mass_distribution, MurataBinned)
    assert recipe.my_updatables is not None
    assert len(recipe.my_updatables) == 1
    assert recipe.my_updatables[0] is recipe.mass_distribution


def test_murata_binned_spec_z_deltasigma_init():
    """Test initialization of MurataBinnedSpecZDeltaSigmaRecipe."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()

    assert recipe is not None
    assert isinstance(recipe, ClusterRecipe)
    assert recipe.integrator is not None
    assert isinstance(recipe.integrator, NumCosmoIntegrator)
    assert recipe.redshift_distribution is not None
    assert isinstance(recipe.redshift_distribution, SpectroscopicRedshift)
    assert recipe.mass_distribution is not None
    assert isinstance(recipe.mass_distribution, MurataBinned)
    assert recipe.my_updatables is not None
    assert len(recipe.my_updatables) == 1
    assert recipe.my_updatables[0] is recipe.mass_distribution


def test_murata_binned_spec_z_get_theory_prediction_no_average():
    """Test get_theory_prediction without averaging."""
    recipe = MurataBinnedSpecZRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_abundance = ClusterAbundance((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_abundance.update_ingredients(cosmo)

    prediction_func = recipe.get_theory_prediction(cluster_abundance, average_on=None)

    mass = np.array([14.0])
    z = np.array([0.5])
    mass_proxy_limits = (0, 5)
    sky_area = 100.0

    result = prediction_func(mass, z, mass_proxy_limits, sky_area)
    assert result is not None
    assert np.all(result > 0)


def test_murata_binned_spec_z_get_theory_prediction_with_mass_average():
    """Test get_theory_prediction with mass averaging."""
    recipe = MurataBinnedSpecZRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_abundance = ClusterAbundance((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_abundance.update_ingredients(cosmo)

    prediction_func = recipe.get_theory_prediction(
        cluster_abundance, average_on=ClusterProperty.MASS
    )

    mass = np.array([14.0])
    z = np.array([0.5])
    mass_proxy_limits = (0, 5)
    sky_area = 100.0

    result = prediction_func(mass, z, mass_proxy_limits, sky_area)
    assert result is not None
    assert np.all(result > 0)


def test_murata_binned_spec_z_get_theory_prediction_with_redshift_average():
    """Test get_theory_prediction with redshift averaging."""
    recipe = MurataBinnedSpecZRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_abundance = ClusterAbundance((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_abundance.update_ingredients(cosmo)

    prediction_func = recipe.get_theory_prediction(
        cluster_abundance, average_on=ClusterProperty.REDSHIFT
    )

    mass = np.array([14.0])
    z = np.array([0.5])
    mass_proxy_limits = (0, 5)
    sky_area = 100.0

    result = prediction_func(mass, z, mass_proxy_limits, sky_area)
    assert result is not None
    assert np.all(result > 0)


def test_murata_binned_spec_z_get_theory_prediction_with_both_average():
    """Test get_theory_prediction with both mass and redshift averaging."""
    recipe = MurataBinnedSpecZRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_abundance = ClusterAbundance((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_abundance.update_ingredients(cosmo)

    prediction_func = recipe.get_theory_prediction(
        cluster_abundance,
        average_on=ClusterProperty.MASS | ClusterProperty.REDSHIFT,
    )

    mass = np.array([14.0])
    z = np.array([0.5])
    mass_proxy_limits = (0, 5)
    sky_area = 100.0

    result = prediction_func(mass, z, mass_proxy_limits, sky_area)
    assert result is not None
    assert np.all(result > 0)


def test_murata_binned_spec_z_get_function_to_integrate():
    """Test get_function_to_integrate method."""
    recipe = MurataBinnedSpecZRecipe()

    mock_prediction = Mock(return_value=np.array([1.0, 2.0]))
    function_mapper = recipe.get_function_to_integrate(mock_prediction)

    int_args = np.array([[14.0, 0.5], [14.5, 0.6]])
    extra_args = np.array([0, 5, 100.0])

    result = function_mapper(int_args, extra_args)
    assert result is not None
    mock_prediction.assert_called_once()


def test_murata_binned_spec_z_evaluate_theory_prediction():
    """Test evaluate_theory_prediction method."""
    recipe = MurataBinnedSpecZRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_abundance = ClusterAbundance((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_abundance.update_ingredients(cosmo)

    mock_bin = Mock(spec=NDimensionalBin)
    mock_bin.mass_proxy_edges = (0, 5)
    mock_bin.z_edges = (0.2, 0.8)

    result = recipe.evaluate_theory_prediction(
        cluster_abundance, mock_bin, 100.0, average_on=None
    )
    assert result > 0


def test_murata_binned_spec_z_deltasigma_get_theory_prediction_no_average():
    """Test get_theory_prediction for DeltaSigma recipe without averaging."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_deltasigma = ClusterDeltaSigma((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_deltasigma.update_ingredients(cosmo)

    prediction_func = recipe.get_theory_prediction(cluster_deltasigma, average_on=None)

    mass = np.array([14.0])
    z = np.array([0.5])
    mass_proxy_limits = (0, 5)
    sky_area = 100.0
    radius_center = 1.5

    with pytest.raises(ValueError, match="The property should be"):
        prediction_func(mass, z, mass_proxy_limits, sky_area, radius_center)


def test_murata_binned_spec_z_deltasigma_get_theory_prediction_with_deltasigma():
    """Test get_theory_prediction for DeltaSigma recipe with DELTASIGMA property."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_deltasigma = ClusterDeltaSigma((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_deltasigma.update_ingredients(cosmo)

    prediction_func = recipe.get_theory_prediction(
        cluster_deltasigma, average_on=ClusterProperty.DELTASIGMA
    )

    mass = np.array([14.0])
    z = np.array([0.5])
    mass_proxy_limits = (0, 5)
    sky_area = 100.0
    radius_center = 1.5

    result = prediction_func(mass, z, mass_proxy_limits, sky_area, radius_center)
    assert result is not None


def test_murata_binned_spec_z_deltasigma_get_function_to_integrate():
    """Test get_function_to_integrate for DeltaSigma recipe."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()

    mock_prediction = Mock(return_value=np.array([1.0, 2.0]))
    function_mapper = recipe.get_function_to_integrate(mock_prediction)

    int_args = np.array([[14.0, 0.5], [14.5, 0.6]])
    extra_args = np.array([0, 5, 100.0, 1.5])

    result = function_mapper(int_args, extra_args)
    assert result is not None
    mock_prediction.assert_called_once()


def test_murata_binned_spec_z_deltasigma_evaluate_theory_prediction():
    """Test evaluate_theory_prediction for DeltaSigma recipe."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_deltasigma = ClusterDeltaSigma((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_deltasigma.update_ingredients(cosmo)

    mock_bin = Mock(spec=NDimensionalBin)
    mock_bin.mass_proxy_edges = (0, 5)
    mock_bin.z_edges = (0.2, 0.8)
    mock_bin.radius_center = 1.5

    result = recipe.evaluate_theory_prediction(
        cluster_deltasigma, mock_bin, 100.0, average_on=ClusterProperty.DELTASIGMA
    )
    assert result > 0


def test_murata_binned_spec_z_deltasigma_get_theory_prediction_counts():
    """Test get_theory_prediction_counts method."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_deltasigma = ClusterDeltaSigma((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_deltasigma.update_ingredients(cosmo)

    prediction_func = recipe.get_theory_prediction_counts(cluster_deltasigma)

    mass = np.array([14.0])
    z = np.array([0.5])
    mass_proxy_limits = (0, 5)
    sky_area = 100.0

    result = prediction_func(mass, z, mass_proxy_limits, sky_area)
    assert result is not None


def test_murata_binned_spec_z_deltasigma_get_function_to_integrate_counts():
    """Test get_function_to_integrate_counts method."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()

    mock_prediction = Mock(return_value=np.array([1.0, 2.0]))
    function_mapper = recipe.get_function_to_integrate_counts(mock_prediction)

    int_args = np.array([[14.0, 0.5], [14.5, 0.6]])
    extra_args = np.array([0, 5, 100.0])

    result = function_mapper(int_args, extra_args)
    assert result is not None
    mock_prediction.assert_called_once()


def test_murata_binned_spec_z_deltasigma_evaluate_theory_prediction_counts():
    """Test evaluate_theory_prediction_counts method."""
    recipe = MurataBinnedSpecZDeltaSigmaRecipe()
    recipe.mass_distribution.mu_p0 = 3.0
    recipe.mass_distribution.mu_p1 = 0.86
    recipe.mass_distribution.mu_p2 = 0.0
    recipe.mass_distribution.sigma_p0 = 3.0
    recipe.mass_distribution.sigma_p1 = 0.7
    recipe.mass_distribution.sigma_p2 = 0.0

    cosmo = pyccl.CosmologyVanillaLCDM()
    hmf = pyccl.halos.MassFuncTinker08()
    cluster_deltasigma = ClusterDeltaSigma((13.0, 17.0), (0.2, 0.8), hmf)
    cluster_deltasigma.update_ingredients(cosmo)

    mock_bin = Mock(spec=NDimensionalBin)
    mock_bin.mass_proxy_edges = (0, 5)
    mock_bin.z_edges = (0.2, 0.8)

    result = recipe.evaluate_theory_prediction_counts(
        cluster_deltasigma, mock_bin, 100.0
    )
    assert result > 0
