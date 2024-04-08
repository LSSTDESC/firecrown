"""
Pytest configuration additions.

Fixtures defined here are available to any test in Firecrown.
"""

from itertools import product
import pytest

import sacc

import numpy as np
import numpy.typing as npt
import pyccl

from firecrown.utils import upper_triangle_indices
from firecrown.likelihood.gauss_family.statistic.statistic import TrivialStatistic
from firecrown.parameters import ParamsMap
from firecrown.connector.mapping import MappingCosmoSIS, mapping_builder
from firecrown.modeling_tools import ModelingTools
from firecrown.metadata.two_point import TracerNames


def pytest_addoption(parser):
    """Add handling of firecrown-specific options for the pytest test runner.

    --runslow: used to run tests marked as slow, which are otherwise not run.
    --integration: used to run only integration tests, which are otherwise not run.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_configure(config):
    """Add new markers that can be set on pytest tests.

    Use the marker `slow` for any test that takes more than a second to run.
    Tests marked as `slow` are not run unless the user requests them by specifying
    the --runslow flag to the pytest program.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Apply our special markers and option handling for pytest."""

    if not config.getoption("--integration"):
        _skip_tests(items, "integration", "need --integration option to run")

    if not config.getoption("--runslow"):
        _skip_tests(items, "slow", "need --runslow option to run")


def _skip_tests(items, keyword, reason):
    """Helper method to skip tests based on a marker name."""

    tests_to_skip = pytest.mark.skip(reason=reason)
    for item in items:
        if keyword in item.keywords:
            item.add_marker(tests_to_skip)


# Fixtures


@pytest.fixture(name="trivial_stats")
def make_stats():
    """Return a non-empty list of TrivialStatistics."""
    return [TrivialStatistic()]


@pytest.fixture(name="trivial_params")
def make_trivial_params() -> ParamsMap:
    """Return a ParamsMap with one parameter."""
    return ParamsMap({"mean": 1.0})


@pytest.fixture(name="sacc_data_for_trivial_stat")
def make_sacc_data():
    """Create a sacc.Sacc object suitable for configuring a
    :class:`TrivialStatistic`."""
    result = sacc.Sacc()
    result.add_data_point("count", (), 1.0)
    result.add_data_point("count", (), 4.0)
    result.add_data_point("count", (), -3.0)
    result.add_covariance([4.0, 9.0, 16.0])
    return result


@pytest.fixture(name="mapping_cosmosis")
def fixture_mapping_cosmosis() -> MappingCosmoSIS:
    """Return a MappingCosmoSIS instance."""
    mapping_cosmosis = mapping_builder(input_style="CosmoSIS")
    assert isinstance(mapping_cosmosis, MappingCosmoSIS)
    mapping_cosmosis.set_params(
        Omega_c=0.26,
        Omega_b=0.04,
        h=0.72,
        A_s=2.1e-9,
        n_s=0.96,
        Omega_k=0.0,
        Neff=3.046,
        m_nu=0.0,
        m_nu_type="normal",
        w0=-1.0,
        wa=0.0,
        T_CMB=2.7255,
    )
    return mapping_cosmosis


# Distribution tests fixtures


@pytest.fixture(name="sacc_with_data_points")
def fixture_sass_missing_covariance() -> sacc.Sacc:
    """Return a Sacc object for configuring a GaussFamily likelihood subclass,
    but which is missing a covariance matrix."""
    result = sacc.Sacc()
    result.add_tracer("misc", "sn_fake_sample")
    for cnt in [7.0, 4.0]:
        result.add_data_point("misc", ("sn_fake_sample",), cnt)
    return result


@pytest.fixture(name="sacc_with_covariance")
def fixture_sacc_with_covariance(sacc_with_data_points: sacc.Sacc) -> sacc.Sacc:
    """Return a Sacc object for configuring a GaussFamily likelihood subclass,
    with a covariance matrix."""
    result = sacc_with_data_points
    cov = np.array([[1.0, -0.5], [-0.5, 1.0]])
    result.add_covariance(cov)
    return result


@pytest.fixture(name="tools_with_vanilla_cosmology")
def fixture_tools_with_vanilla_cosmology():
    """Return a ModelingTools object containing the LCDM cosmology from
    pyccl."""
    result = ModelingTools()
    result.update(ParamsMap())
    result.prepare(pyccl.CosmologyVanillaLCDM())


@pytest.fixture(name="cluster_sacc_data")
def fixture_cluster_sacc_data() -> sacc.Sacc:
    # pylint: disable=no-member
    cc = sacc.standard_types.cluster_counts
    # pylint: disable=no-member
    mlm = sacc.standard_types.cluster_mean_log_mass

    s = sacc.Sacc()
    s.add_tracer("survey", "my_survey", 4000)
    s.add_tracer("survey", "not_my_survey", 4000)
    s.add_tracer("bin_z", "z_bin_tracer_1", 0, 2)
    s.add_tracer("bin_z", "z_bin_tracer_2", 2, 4)
    s.add_tracer("bin_richness", "mass_bin_tracer_1", 0, 2)
    s.add_tracer("bin_richness", "mass_bin_tracer_2", 2, 4)

    s.add_data_point(cc, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_1"), 1)
    s.add_data_point(cc, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_2"), 1)
    s.add_data_point(cc, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_1"), 1)
    s.add_data_point(cc, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_2"), 1)

    s.add_data_point(mlm, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_1"), 1)
    s.add_data_point(mlm, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_2"), 1)
    s.add_data_point(mlm, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_1"), 1)
    s.add_data_point(mlm, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_2"), 1)

    return s


# Two-point related SACC data fixtures


@pytest.fixture(name="sacc_galaxy_cells_src0_src0")
def fixture_sacc_galaxy_cells_src0_src0():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_shear_cl_ee", "src0", "src0", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz


@pytest.fixture(name="sacc_galaxy_cells_src0_src1")
def fixture_sacc_galaxy_cells_src0_src1():
    """Fixture for a SACC data without window functions."""

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    sacc_data = sacc.Sacc()

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.7) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src1", z, dndz1)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_shear_cl_ee", "src0", "src1", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_cells_lens0_lens0")
def fixture_sacc_galaxy_cells_lens0_lens0():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_density_cl", "lens0", "lens0", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz


@pytest.fixture(name="sacc_galaxy_cells_lens0_lens1")
def fixture_sacc_galaxy_cells_lens0_lens1():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz0 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz0)
    dndz1 = np.exp(-0.5 * (z - 0.6) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens1", z, dndz1)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_density_cl", "lens0", "lens1", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_xis_lens0_lens0")
def fixture_sacc_galaxy_xis_lens0_lens0():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz)

    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi("galaxy_density_xi", "lens0", "lens0", thetas, xis)

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz


@pytest.fixture(name="sacc_galaxy_xis_lens0_lens1")
def fixture_sacc_galaxy_xis_lens0_lens1():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz0 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz0)
    dndz1 = np.exp(-0.5 * (z - 0.6) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens1", z, dndz1)

    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi("galaxy_density_xi", "lens0", "lens1", thetas, xis)

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_cells_src0_lens0")
def fixture_sacc_galaxy_cells_src0_lens0():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz1)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_shearDensity_cl_e", "src0", "lens0", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_xis_src0_lens0")
def fixture_sacc_galaxy_xis_src0_lens0():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz1)

    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi("galaxy_shearDensity_xi_t", "src0", "lens0", thetas, xis)

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_cells")
def fixture_sacc_galaxy_cells():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    src_bins_centers = np.linspace(0.25, 0.75, 5)
    lens_bins_centers = np.linspace(0.1, 0.6, 5)

    tracers: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    tracer_pairs: dict[
        TracerNames, tuple[str, npt.NDArray[np.int64], npt.NDArray[np.float64]]
    ] = {}

    for i, mn in enumerate(src_bins_centers):
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.05 / 0.05)
        sacc_data.add_tracer("NZ", f"src{i}", z, dndz)
        tracers[f"src{i}"] = (z, dndz)

    for i, mn in enumerate(lens_bins_centers):
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.05 / 0.05)
        sacc_data.add_tracer("NZ", f"lens{i}", z, dndz)
        tracers[f"lens{i}"] = (z, dndz)

    dv = []

    for i, j in upper_triangle_indices(len(src_bins_centers)):
        Cells = np.random.normal(size=ells.shape[0])
        sacc_data.add_ell_cl("galaxy_shear_cl_ee", f"src{i}", f"src{j}", ells, Cells)
        tracer_pairs[TracerNames(f"src{i}", f"src{j}")] = (
            "galaxy_shear_cl_ee",
            ells,
            Cells,
        )
        dv.append(Cells)

    for i, j in upper_triangle_indices(len(lens_bins_centers)):
        Cells = np.random.normal(size=ells.shape[0])
        sacc_data.add_ell_cl("galaxy_density_cl", f"lens{i}", f"lens{j}", ells, Cells)
        tracer_pairs[TracerNames(f"lens{i}", f"lens{j}")] = (
            "galaxy_density_cl",
            ells,
            Cells,
        )
        dv.append(Cells)

    for i, j in product(range(len(src_bins_centers)), range(len(lens_bins_centers))):
        Cells = np.random.normal(size=ells.shape[0])
        sacc_data.add_ell_cl(
            "galaxy_shearDensity_cl_e", f"src{i}", f"lens{j}", ells, Cells
        )
        tracer_pairs[TracerNames(f"src{i}", f"lens{j}")] = (
            "galaxy_shearDensity_cl_e",
            ells,
            Cells,
        )
        dv.append(Cells)

    delta_v = np.concatenate(dv, axis=0)
    cov = np.diag(np.ones_like(delta_v) * 0.01)

    sacc_data.add_covariance(cov)

    return sacc_data, tracers, tracer_pairs


@pytest.fixture(name="sacc_galaxy_xis")
def fixture_sacc_galaxy_xis():
    """Fixture for a SACC data without window functions."""

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    sacc_data = sacc.Sacc()

    src_bins_centers = np.linspace(0.25, 0.75, 5)
    lens_bins_centers = np.linspace(0.1, 0.6, 5)

    tracers: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    tracer_pairs: dict[
        TracerNames, tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ] = {}

    for i, mn in enumerate(src_bins_centers):
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.05 / 0.05)
        sacc_data.add_tracer("NZ", f"src{i}", z, dndz)
        tracers[f"src{i}"] = (z, dndz)

    for i, mn in enumerate(lens_bins_centers):
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.05 / 0.05)
        sacc_data.add_tracer("NZ", f"lens{i}", z, dndz)
        tracers[f"lens{i}"] = (z, dndz)

    dv = []

    for i, j in upper_triangle_indices(len(lens_bins_centers)):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi("galaxy_density_xi", f"lens{i}", f"lens{j}", thetas, xis)
        tracer_pairs[TracerNames(f"lens{i}", f"lens{j}")] = (
            "galaxy_density_xi",
            thetas,
            xis,
        )
        dv.append(xis)

    for i, j in product(range(len(src_bins_centers)), range(len(lens_bins_centers))):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi(
            "galaxy_shearDensity_xi_t", f"src{i}", f"lens{j}", thetas, xis
        )
        tracer_pairs[TracerNames(f"src{i}", f"lens{j}")] = (
            "galaxy_shearDensity_xi_t",
            thetas,
            xis,
        )
        dv.append(xis)

    delta_v = np.concatenate(dv, axis=0)
    cov = np.diag(np.ones_like(delta_v) * 0.01)

    sacc_data.add_covariance(cov)

    return sacc_data, tracers, tracer_pairs
