"""Tests for the cluster delta sigma module."""

from unittest.mock import Mock

import numpy as np
import pyccl
import pytest

from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.deltasigma import ClusterDeltaSigma
from firecrown.models.cluster.binning import NDimensionalBin
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.kernel import SpectroscopicRedshift
from firecrown.models.cluster.mass_proxy import MurataBinned
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe
from firecrown.models.cluster.recipes.murata_binned_spec_z import (
    MurataBinnedSpecZRecipe,
)
from firecrown.models.cluster.recipes.murata_binned_spec_z_deltasigma import (
    MurataBinnedSpecZDeltaSigmaRecipe,
)


@pytest.fixture(name="cluster_abundance")
def fixture_cluster_abundance() -> ClusterAbundance:
    hmf = pyccl.halos.MassFuncBocquet16(mass_def="200c")
    cl_abundance = ClusterAbundance(
        min_z=0,
        max_z=2,
        min_mass=13,
        max_mass=17,
        halo_mass_function=hmf,
    )
    cl_abundance.update_ingredients(pyccl.CosmologyVanillaLCDM())
    return cl_abundance


@pytest.fixture(name="cluster_deltasigma")
def fixture_cluster_deltasigma() -> ClusterDeltaSigma:
    hmf = pyccl.halos.MassFuncBocquet16(mass_def="200c")
    cl_deltasigma = ClusterDeltaSigma(
        min_z=0,
        max_z=2,
        min_mass=13,
        max_mass=17,
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
    cluster_abundance: ClusterAbundance,
    cluster_deltasigma: ClusterDeltaSigma,
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZDeltaSigmaRecipe,
):
    prediction = murata_binned_spec_z_deltasigma.get_theory_prediction(
        cluster_abundance, cluster_deltasigma, ClusterProperty.DELTASIGMA
    )

    assert prediction is not None
    assert callable(prediction)

    mass = np.linspace(13, 17, 2)
    z = np.linspace(0.1, 1, 2)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2
    radius_center = 1.5

    result = prediction(mass, z, mass_proxy_limits, sky_area, radius_center)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)


def test_get_function_to_integrate_returns_value(
    cluster_abundance: ClusterAbundance,
    cluster_deltasigma: ClusterDeltaSigma,
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZDeltaSigmaRecipe,
):
    prediction = murata_binned_spec_z_deltasigma.get_theory_prediction(
        cluster_abundance, cluster_deltasigma, ClusterProperty.DELTASIGMA
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


def test_evaluates_theory_prediction_returns_value(
    cluster_abundance: ClusterAbundance,
    cluster_deltasigma: ClusterDeltaSigma,
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZDeltaSigmaRecipe,
):
    mock_bin = Mock(spec=NDimensionalBin)
    mock_bin.mass_proxy_edges = (0, 5)
    mock_bin.z_edges = (0, 1)
    mock_bin.radius_center = 1.5
    average_on = ClusterProperty.DELTASIGMA

    prediction = murata_binned_spec_z_deltasigma.evaluate_theory_prediction(
        [cluster_abundance, cluster_deltasigma], mock_bin, 360**2, average_on
    )

    assert prediction > 0
