"""Unit tests for firecrown.app.examples._cluster_data module."""

import numpy as np
import pytest
import sacc
from numcosmo_py import Nc

from firecrown.app.analysis import CCLCosmologySpec
from firecrown.app.examples._cluster_data import (
    FIDUCIAL_OMEGA_C,
    FIDUCIAL_SIGMA8,
    PROFILE_N_RADIUS_BINS,
    SDSS_SURVEY_NAME,
    SIM_MIN_LN_RICHNESS,
    SIM_N_RICHNESS_BINS,
    SIM_N_Z_BINS,
    SIM_Z_RANGE,
    BinnedClusters,
    ClusterCatalog,
    bin_catalog,
    counts_sacc,
    fiducial_cosmology_spec,
    sdss_sacc,
    shear_sacc,
    simulate_catalog,
    stack_profiles,
)

# pylint: disable=no-member
COUNTS = sacc.standard_types.cluster_counts
MEAN_LOG_MASS = sacc.standard_types.cluster_mean_log_mass
DELTA_SIGMA = sacc.standard_types.cluster_delta_sigma
SHEAR = sacc.standard_types.cluster_shear
# pylint: enable=no-member

N_BINS = SIM_N_Z_BINS * SIM_N_RICHNESS_BINS


def make_catalog(n_clusters: int = 400, seed: int = 0) -> ClusterCatalog:
    """Create a small catalog filling every redshift-richness bin."""
    rng = np.random.default_rng(seed)
    return ClusterCatalog(
        z=rng.uniform(*SIM_Z_RANGE, n_clusters),
        log10_richness=rng.uniform(1.0, 2.0, n_clusters),
        log10_mass=rng.normal(14.0, 0.3, n_clusters),
    )


@pytest.fixture(name="catalog", scope="module")
def fixture_catalog() -> ClusterCatalog:
    """Small synthetic catalog."""
    return make_catalog()


@pytest.fixture(name="binned", scope="module")
def fixture_binned(catalog: ClusterCatalog) -> BinnedClusters:
    """Binned synthetic catalog."""
    return bin_catalog(catalog)


def test_fiducial_cosmology_spec() -> None:
    """Only Omega_c and sigma8 are free, at the simulation values."""
    spec = fiducial_cosmology_spec()

    assert isinstance(spec, CCLCosmologySpec)
    free = {p.name for p in spec.parameters if p.free}
    assert free == {"Omega_c", "sigma8"}
    assert spec["Omega_c"].default_value == FIDUCIAL_OMEGA_C
    assert spec["sigma8"].default_value == FIDUCIAL_SIGMA8
    assert spec["h"].default_value == pytest.approx(0.71)


def test_simulate_catalog_is_reproducible() -> None:
    """The same seed gives the same catalog, within the selection cuts."""
    first = simulate_catalog(32, Nc.MultiplicityFuncMassDef.MEAN)
    second = simulate_catalog(32, Nc.MultiplicityFuncMassDef.MEAN)

    assert len(first.z) > 0
    np.testing.assert_array_equal(first.z, second.z)
    np.testing.assert_array_equal(first.log10_mass, second.log10_mass)
    assert np.all(first.z >= SIM_Z_RANGE[0])
    assert np.all(first.z <= SIM_Z_RANGE[1])
    assert np.all(first.log10_richness > SIM_MIN_LN_RICHNESS / np.log(10.0))


def test_bin_catalog(catalog: ClusterCatalog, binned: BinnedClusters) -> None:
    """Every cluster is counted once and the bins have the expected shape."""
    assert binned.counts.shape == (SIM_N_Z_BINS, SIM_N_RICHNESS_BINS)
    assert binned.counts.sum() == len(catalog.z)
    assert len(binned.z_edges) == SIM_N_Z_BINS + 1
    assert len(binned.richness_edges) == SIM_N_RICHNESS_BINS + 1
    assert np.all(np.isfinite(binned.mean_log10_mass))
    assert np.all(binned.var_mean_log10_mass > 0.0)


def test_counts_sacc(binned: BinnedClusters) -> None:
    """The counts SACC holds counts and mean masses with a matching covariance."""
    sacc_data = counts_sacc("survey", binned)

    assert len(sacc_data.get_data_points(COUNTS)) == N_BINS
    assert len(sacc_data.get_data_points(MEAN_LOG_MASS)) == N_BINS
    assert sacc_data.covariance.dense.shape == (2 * N_BINS, 2 * N_BINS)
    assert "survey" in sacc_data.tracers


def test_stack_profiles_and_shear_sacc(
    catalog: ClusterCatalog, binned: BinnedClusters
) -> None:
    """Profiles are stacked per bin and stored with all four observables."""
    profiles = stack_profiles(
        catalog, binned, fiducial_cosmology_spec().to_ccl_cosmology()
    )

    shape = (SIM_N_Z_BINS, SIM_N_RICHNESS_BINS, PROFILE_N_RADIUS_BINS)
    assert profiles.mean_delta_sigma.shape == shape
    assert profiles.mean_reduced_shear.shape == shape
    assert len(profiles.radius_edges) == PROFILE_N_RADIUS_BINS + 1
    assert np.all(profiles.mean_delta_sigma > 0.0)
    assert np.all(profiles.mean_reduced_shear > 0.0)

    sacc_data = shear_sacc("survey", binned, profiles)
    n_profile = N_BINS * PROFILE_N_RADIUS_BINS
    assert len(sacc_data.get_data_points(DELTA_SIGMA)) == n_profile
    assert len(sacc_data.get_data_points(SHEAR)) == n_profile
    n_total = 2 * N_BINS + 2 * n_profile
    assert sacc_data.covariance.dense.shape == (n_total, n_total)


def test_sdss_sacc() -> None:
    """The SDSS data has one redshift bin and five richness bins."""
    sacc_data = sdss_sacc()

    assert SDSS_SURVEY_NAME in sacc_data.tracers
    counts = sacc_data.get_mean(COUNTS)
    np.testing.assert_array_equal(counts, [3711, 1788, 978, 476, 223])
    assert len(sacc_data.get_data_points(MEAN_LOG_MASS)) == 5
    assert sacc_data.covariance.dense.shape == (10, 10)
