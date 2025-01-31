"""Tests for the cluster abundance module."""

from unittest.mock import Mock

import numpy as np
import pyccl
import pytest

from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.binning import NDimensionalBin
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.kernel import SpectroscopicRedshift
from firecrown.models.cluster.mass_proxy import MurataBinned
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe
from firecrown.models.cluster.recipes.murata_binned_spec_z_selection import (
    MurataBinnedSpecZSelectionRecipe,
)


@pytest.fixture(name="cluster_abundance")
def fixture_cluster_abundance() -> ClusterAbundance:
    hmf = pyccl.halos.MassFuncBocquet16()
    cl_abundance = ClusterAbundance(
        min_z=0,
        max_z=2,
        min_mass=13,
        max_mass=17,
        halo_mass_function=hmf,
    )
    cl_abundance.update_ingredients(pyccl.CosmologyVanillaLCDM())
    return cl_abundance


@pytest.fixture(name="murata_binned_spec_z_selection")
def fixture_murata_binned_spec_z() -> MurataBinnedSpecZSelectionRecipe:
    cluster_recipe = MurataBinnedSpecZSelectionRecipe()
    cluster_recipe.mass_distribution.mu_p0 = 3.0
    cluster_recipe.mass_distribution.mu_p1 = 0.86
    cluster_recipe.mass_distribution.mu_p2 = 0.0
    cluster_recipe.mass_distribution.sigma_p0 = 3.0
    cluster_recipe.mass_distribution.sigma_p1 = 0.7
    cluster_recipe.mass_distribution.sigma_p2 = 0.0
    cluster_recipe.purity_distribution.ap_rc = 1.1839
    cluster_recipe.purity_distribution.bp_rc = -0.4077
    cluster_recipe.purity_distribution.ap_nc = 3.9193
    cluster_recipe.purity_distribution.bp_nc = -0.3323
    cluster_recipe.completeness_distribution.ac_mc = 13.31
    cluster_recipe.completeness_distribution.bc_mc = 0.2025
    cluster_recipe.completeness_distribution.ac_nc = 0.38
    cluster_recipe.completeness_distribution.bc_nc = 1.2634
    return cluster_recipe


def test_murata_binned_spec_z_init():
    recipe = MurataBinnedSpecZSelectionRecipe()

    assert recipe is not None
    assert isinstance(recipe, ClusterRecipe)
    assert recipe.integrator is not None
    assert isinstance(recipe.integrator, NumCosmoIntegrator)
    assert recipe.redshift_distribution is not None
    assert isinstance(recipe.redshift_distribution, SpectroscopicRedshift)
    assert recipe.mass_distribution is not None
    assert isinstance(recipe.mass_distribution, MurataBinned)
    assert recipe.completeness_distribution is not None
    assert recipe.purity_distribution is not None
    assert recipe.my_updatables is not None
    assert len(recipe.my_updatables) == 3
    assert recipe.my_updatables[0] is recipe.mass_distribution
    assert recipe.my_updatables[1] is recipe.completeness_distribution
    assert recipe.my_updatables[2] is recipe.purity_distribution


def test_get_theory_prediction_returns_value(
    cluster_abundance: ClusterAbundance,
    murata_binned_spec_z_selection: MurataBinnedSpecZSelectionRecipe,
):
    prediction = murata_binned_spec_z_selection.get_theory_prediction(cluster_abundance)

    assert prediction is not None
    assert callable(prediction)

    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)


def test_get_theory_prediction_with_average_returns_value(
    cluster_abundance: ClusterAbundance,
    murata_binned_spec_z_selection: MurataBinnedSpecZSelectionRecipe,
):
    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2

    prediction = murata_binned_spec_z_selection.get_theory_prediction(
        cluster_abundance, average_on=ClusterProperty.MASS
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction = murata_binned_spec_z_selection.get_theory_prediction(
        cluster_abundance, average_on=ClusterProperty.REDSHIFT
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction = murata_binned_spec_z_selection.get_theory_prediction(
        cluster_abundance, average_on=(ClusterProperty.REDSHIFT | ClusterProperty.MASS)
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)


def test_get_theory_prediction_throws_with_nonimpl_average(
    cluster_abundance: ClusterAbundance,
    murata_binned_spec_z_selection: MurataBinnedSpecZSelectionRecipe,
):
    prediction = murata_binned_spec_z_selection.get_theory_prediction(
        cluster_abundance, average_on=ClusterProperty.SHEAR
    )

    assert prediction is not None
    assert callable(prediction)

    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2

    with pytest.raises(NotImplementedError):
        _ = prediction(mass, z, mass_proxy_limits, sky_area)


def test_get_function_to_integrate_returns_value(
    cluster_abundance: ClusterAbundance,
    murata_binned_spec_z_selection: MurataBinnedSpecZSelectionRecipe,
):
    prediction = murata_binned_spec_z_selection.get_theory_prediction(cluster_abundance)
    function_to_integrate = murata_binned_spec_z_selection.get_function_to_integrate(
        prediction
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
    cluster_abundance: ClusterAbundance,
    murata_binned_spec_z_selection: MurataBinnedSpecZSelectionRecipe,
):
    mock_bin = Mock(spec=NDimensionalBin)
    mock_bin.mass_proxy_edges = (0, 5)
    mock_bin.z_edges = (0, 1)

    prediction = murata_binned_spec_z_selection.evaluate_theory_prediction(
        cluster_abundance, mock_bin, 360**2
    )

    assert prediction > 0
