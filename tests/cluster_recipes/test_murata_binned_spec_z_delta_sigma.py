"""Tests for the cluster delta sigma module."""

from unittest.mock import Mock

import numpy as np
import pyccl
import pytest

from firecrown.models.cluster import (
    ClusterDeltaSigma,
    ClusterProperty,
    MurataBinned,
    NDimensionalBin,
    SpectroscopicRedshift,
)
from firecrown.models.cluster import NumCosmoIntegrator
from firecrown.models.cluster import (
    ClusterRecipe,
    MurataBinnedSpecZDeltaSigmaRecipe,
    MurataBinnedSpecZRecipe,
)


@pytest.fixture(name="cluster_deltasigma")
def fixture_cluster_deltasigma() -> ClusterDeltaSigma:
    hmf = pyccl.halos.MassFuncBocquet16(mass_def="200c")
    cl_deltasigma = ClusterDeltaSigma(
        z_interval=(0, 2),
        mass_interval=(13, 17),
        halo_mass_function=hmf,
    )
    cl_deltasigma.update_ingredients(pyccl.CosmologyVanillaLCDM())
    return cl_deltasigma


@pytest.fixture(name="murata_binned_spec_z")
def fixture_murata_binned_spec_z() -> MurataBinnedSpecZRecipe:
    cluster_recipe = MurataBinnedSpecZRecipe()
    cluster_recipe.mass_distribution.mu_p0 = 3.0
    cluster_recipe.mass_distribution.mu_p1 = 0.86
    cluster_recipe.mass_distribution.mu_p2 = 0.0
    cluster_recipe.mass_distribution.sigma_p0 = 3.0
    cluster_recipe.mass_distribution.sigma_p1 = 0.7
    cluster_recipe.mass_distribution.sigma_p2 = 0.0
    return cluster_recipe


@pytest.fixture(name="murata_binned_spec_z_deltasigma")
def fixture_murata_binned_spec_z_deltasigma() -> MurataBinnedSpecZDeltaSigmaRecipe:
    cluster_recipe = MurataBinnedSpecZDeltaSigmaRecipe()
    cluster_recipe.mass_distribution.mu_p0 = 3.0
    cluster_recipe.mass_distribution.mu_p1 = 0.86
    cluster_recipe.mass_distribution.mu_p2 = 0.0
    cluster_recipe.mass_distribution.sigma_p0 = 3.0
    cluster_recipe.mass_distribution.sigma_p1 = 0.7
    cluster_recipe.mass_distribution.sigma_p2 = 0.0
    return cluster_recipe


def test_murata_binned_spec_z_deltasigma_init():
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


def test_get_theory_prediction_returns_value(
    cluster_deltasigma: ClusterDeltaSigma,
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZDeltaSigmaRecipe,
):
    prediction_none = murata_binned_spec_z_deltasigma.get_theory_prediction(
        cluster_deltasigma, average_on=None
    )
    prediction = murata_binned_spec_z_deltasigma.get_theory_prediction(
        cluster_deltasigma, ClusterProperty.DELTASIGMA
    )
    prediction_c = murata_binned_spec_z_deltasigma.get_theory_prediction_counts(
        cluster_deltasigma
    )

    assert prediction is not None
    assert prediction_c is not None
    assert callable(prediction)
    assert callable(prediction_c)

    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2
    radius_center = 1.5
    with pytest.raises(
        ValueError,
        match=f"The property should be" f" {ClusterProperty.DELTASIGMA}.",
    ):
        result = prediction_none(mass, z, mass_proxy_limits, sky_area, radius_center)

    result = prediction(mass, z, mass_proxy_limits, sky_area, radius_center)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    result_c = prediction_c(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result_c, np.ndarray)
    assert np.issubdtype(result_c.dtype, np.float64)
    assert len(result_c) == 2
    assert np.all(result_c > 0)


def test_get_function_to_integrate_returns_value(
    cluster_deltasigma: ClusterDeltaSigma,
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZDeltaSigmaRecipe,
):
    prediction = murata_binned_spec_z_deltasigma.get_theory_prediction(
        cluster_deltasigma, ClusterProperty.DELTASIGMA
    )
    function_to_integrate = murata_binned_spec_z_deltasigma.get_function_to_integrate(
        prediction
    )

    assert function_to_integrate is not None
    assert callable(function_to_integrate)

    int_args = np.array([[13.0, 0.1], [17.0, 1.0]])
    extra_args = np.array([0, 5, 360**2, 1.5])

    result = function_to_integrate(int_args, extra_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction_c = murata_binned_spec_z_deltasigma.get_theory_prediction_counts(
        cluster_deltasigma
    )
    function_to_integrate = (
        murata_binned_spec_z_deltasigma.get_function_to_integrate_counts(prediction_c)
    )

    assert function_to_integrate is not None
    assert callable(function_to_integrate)

    int_args = np.array([[13.0, 0.1], [17.0, 1.0]])
    extra_args = np.array([0, 5, 360**2])

    result = function_to_integrate(int_args, extra_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)


def test_evaluates_theory_prediction_returns_value(
    cluster_deltasigma: ClusterDeltaSigma,
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZDeltaSigmaRecipe,
):
    mock_bin = Mock(spec=NDimensionalBin)
    mock_bin.mass_proxy_edges = (2, 5)
    mock_bin.z_edges = (0.5, 1)
    mock_bin.radius_center = 1.5
    average_on = ClusterProperty.DELTASIGMA

    prediction = murata_binned_spec_z_deltasigma.evaluate_theory_prediction(
        cluster_deltasigma, mock_bin, 360**2, average_on
    )
    prediction_c = murata_binned_spec_z_deltasigma.evaluate_theory_prediction_counts(
        cluster_deltasigma, mock_bin, 360**2
    )
    assert prediction > 0
    assert prediction_c > 0
