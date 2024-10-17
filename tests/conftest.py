"""
Pytest configuration additions.

Fixtures defined here are available to any test in Firecrown.
"""

from itertools import product
import pytest

import sacc

import numpy as np
import numpy.typing as npt

from firecrown.updatable import get_default_params_map
from firecrown.utils import upper_triangle_indices
from firecrown.likelihood.statistic import TrivialStatistic
from firecrown.parameters import ParamsMap
from firecrown.connector.mapping import MappingCosmoSIS, mapping_builder
from firecrown.modeling_tools import ModelingTools
from firecrown.metadata_types import (
    Galaxies,
    InferredGalaxyZDist,
    TracerNames,
    TwoPointHarmonic,
    TwoPointXY,
    TwoPointReal,
    measurement_is_compatible_harmonic,
    measurement_is_compatible_real,
)
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc


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
    params = get_default_params_map(result)
    result.update(params)
    result.prepare()

    return result


@pytest.fixture(
    name="harmonic_bin_1",
    params=[Galaxies.COUNTS, Galaxies.SHEAR_E],
    ids=["counts", "shear_e"],
)
def make_harmonic_bin_1(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 5 bins."""
    z = np.linspace(0.0, 1.0, 256)  # Necessary to match the default lensing kernel size
    z_mean = 0.5
    z_sigma = 0.05
    dndz = np.exp(-0.5 * (z - z_mean) ** 2 / z_sigma**2) / (
        np.sqrt(2 * np.pi) * z_sigma
    )
    x = InferredGalaxyZDist(
        bin_name="bin_1", z=z, dndz=dndz, measurements={request.param}
    )
    return x


@pytest.fixture(
    name="harmonic_bin_2",
    params=[Galaxies.COUNTS, Galaxies.SHEAR_E],
    ids=["counts", "shear_e"],
)
def make_harmonic_bin_2(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 3 bins."""
    z = np.linspace(0.0, 1.0, 256)  # Necessary to match the default lensing kernel size
    z_mean = 0.6
    z_sigma = 0.05
    dndz = np.exp(-0.5 * (z - z_mean) ** 2 / z_sigma**2) / (
        np.sqrt(2 * np.pi) * z_sigma
    )
    x = InferredGalaxyZDist(
        bin_name="bin_2", z=z, dndz=dndz, measurements={request.param}
    )
    return x


@pytest.fixture(
    name="real_bin_1",
    params=[
        Galaxies.COUNTS,
        Galaxies.SHEAR_T,
        Galaxies.SHEAR_MINUS,
        Galaxies.SHEAR_PLUS,
    ],
    ids=["counts", "shear_t", "shear_minus", "shear_plus"],
)
def make_real_bin_1(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 5 bins."""
    x = InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.linspace(0, 1, 5),
        dndz=np.array([0.1, 0.5, 0.2, 0.3, 0.4]),
        measurements={request.param},
    )
    return x


@pytest.fixture(
    name="real_bin_2",
    params=[
        Galaxies.COUNTS,
        Galaxies.SHEAR_T,
        Galaxies.SHEAR_MINUS,
        Galaxies.SHEAR_PLUS,
    ],
    ids=["counts", "shear_t", "shear_minus", "shear_plus"],
)
def make_real_bin_2(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 3 bins."""
    x = InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.linspace(0, 1, 3),
        dndz=np.array([0.1, 0.5, 0.4]),
        measurements={request.param},
    )
    return x


@pytest.fixture(name="window_1")
def make_window_1() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Generate a Window object with 100 ells."""
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

    return ells, weights


@pytest.fixture(name="harmonic_two_point_xy")
def make_harmonic_two_point_xy(
    harmonic_bin_1: InferredGalaxyZDist,
    harmonic_bin_2: InferredGalaxyZDist,
) -> TwoPointXY:
    """Generate a TwoPointCWindow object with 100 ells."""
    m1 = list(harmonic_bin_1.measurements)[0]
    m2 = list(harmonic_bin_2.measurements)[0]
    if not measurement_is_compatible_harmonic(m1, m2):
        pytest.skip("Incompatible measurements")
    xy = TwoPointXY(
        x=harmonic_bin_1, y=harmonic_bin_2, x_measurement=m1, y_measurement=m2
    )
    return xy


@pytest.fixture(name="real_two_point_xy")
def make_real_two_point_xy(
    real_bin_1: InferredGalaxyZDist,
    real_bin_2: InferredGalaxyZDist,
) -> TwoPointXY:
    """Generate a TwoPointCWindow object with 100 ells."""
    m1 = list(real_bin_1.measurements)[0]
    m2 = list(real_bin_2.measurements)[0]
    if not measurement_is_compatible_real(m1, m2):
        pytest.skip("Incompatible measurements")
    xy = TwoPointXY(x=real_bin_1, y=real_bin_2, x_measurement=m1, y_measurement=m2)
    return xy


@pytest.fixture(name="two_point_cwindow")
def make_two_point_cwindow(
    window_1: tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]],
    harmonic_two_point_xy: TwoPointXY,
) -> TwoPointHarmonic:
    """Generate a TwoPointCWindow object with 100 ells."""
    two_point = TwoPointHarmonic(
        XY=harmonic_two_point_xy, ells=window_1[0], window=window_1[1]
    )
    return two_point


@pytest.fixture(name="two_point_cell")
def make_two_point_cell(
    harmonic_two_point_xy: TwoPointXY,
) -> TwoPointHarmonic:
    """Generate a TwoPointCWindow object with 100 ells."""
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    return TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy)


@pytest.fixture(name="two_point_real")
def make_two_point_real(real_two_point_xy: TwoPointXY) -> TwoPointReal:
    """Generate a TwoPointCWindow object with 100 ells."""
    thetas = np.array(np.linspace(0, 100, 100), dtype=np.float64)
    return TwoPointReal(thetas=thetas, XY=real_two_point_xy)


@pytest.fixture(name="cluster_sacc_data")
def fixture_cluster_sacc_data() -> sacc.Sacc:
    """Return a Sacc object with cluster data."""
    # pylint: disable=no-member
    cc = sacc.standard_types.cluster_counts
    # pylint: disable=no-member
    mlm = sacc.standard_types.cluster_mean_log_mass
    # pylint: disable=no-member
    cs = sacc.standard_types.cluster_shear

    s = sacc.Sacc()
    s.add_tracer("survey", "my_survey", 4000)
    s.add_tracer("survey", "not_my_survey", 4000)
    s.add_tracer("bin_z", "z_bin_tracer_1", 0, 2)
    s.add_tracer("bin_z", "z_bin_tracer_2", 2, 4)
    s.add_tracer("bin_richness", "mass_bin_tracer_1", 0, 2)
    s.add_tracer("bin_richness", "mass_bin_tracer_2", 2, 4)
    s.add_tracer("bin_radius", "radius_bin_tracer_1", 0, 2, 1)
    s.add_tracer("bin_radius", "radius_bin_tracer_2", 2, 4, 3)

    s.add_data_point(cc, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_1"), 1)
    s.add_data_point(cc, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_2"), 1)
    s.add_data_point(cc, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_1"), 1)
    s.add_data_point(cc, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_2"), 1)

    s.add_data_point(mlm, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_1"), 1)
    s.add_data_point(mlm, ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_2"), 1)
    s.add_data_point(mlm, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_1"), 1)
    s.add_data_point(mlm, ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_2"), 1)
    s.add_data_point(
        cs,
        ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_1", "radius_bin_tracer_1"),
        1,
    )
    s.add_data_point(
        cs,
        ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_1", "radius_bin_tracer_2"),
        1,
    )
    s.add_data_point(
        cs,
        ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_1", "radius_bin_tracer_1"),
        1,
    )
    s.add_data_point(
        cs,
        ("not_my_survey", "z_bin_tracer_2", "mass_bin_tracer_1", "radius_bin_tracer_2"),
        1,
    )

    return s


# Two-point related SACC data fixtures


@pytest.fixture(name="sacc_galaxy_cells_src0_src0")
def fixture_sacc_galaxy_cells_src0_src0():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05
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

    z = np.linspace(0, 1.0, 256) + 0.05
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

    z = np.linspace(0, 1.0, 256) + 0.05
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

    z = np.linspace(0, 1.0, 256) + 0.05
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

    z = np.linspace(0, 1.0, 256) + 0.05
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

    z = np.linspace(0, 1.0, 256) + 0.05
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

    z = np.linspace(0, 1.0, 256) + 0.05
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

    z = np.linspace(0, 1.0, 256) + 0.05
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
def fixture_sacc_galaxy_cells() -> tuple[sacc.Sacc, dict, dict]:
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    src_bins_centers = np.linspace(0.25, 0.75, 5)
    lens_bins_centers = np.linspace(0.1, 0.6, 5)

    tracers: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    tracer_pairs: dict[
        tuple[TracerNames, str], tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]
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
        tracer_pairs[(TracerNames(f"src{i}", f"src{j}"), "galaxy_shear_cl_ee")] = (
            ells,
            Cells,
        )
        dv.append(Cells)

    for i, j in upper_triangle_indices(len(lens_bins_centers)):
        Cells = np.random.normal(size=ells.shape[0])
        sacc_data.add_ell_cl("galaxy_density_cl", f"lens{i}", f"lens{j}", ells, Cells)
        tracer_pairs[(TracerNames(f"lens{i}", f"lens{j}"), "galaxy_density_cl")] = (
            ells,
            Cells,
        )
        dv.append(Cells)

    for i, j in product(range(len(src_bins_centers)), range(len(lens_bins_centers))):
        Cells = np.random.normal(size=ells.shape[0])
        sacc_data.add_ell_cl(
            "galaxy_shearDensity_cl_e", f"src{i}", f"lens{j}", ells, Cells
        )
        tracer_pairs[
            (TracerNames(f"src{i}", f"lens{j}"), "galaxy_shearDensity_cl_e")
        ] = (
            ells,
            Cells,
        )
        dv.append(Cells)

    delta_v = np.concatenate(dv, axis=0)
    cov = np.diag(np.ones_like(delta_v) * 0.01)

    sacc_data.add_covariance(cov)

    return sacc_data, tracers, tracer_pairs


@pytest.fixture(name="sacc_galaxy_cwindows")
def fixture_sacc_galaxy_cwindows():
    """Fixture for a SACC data with window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))
    nobs = len(ells) - 5

    src_bins_centers = np.linspace(0.25, 0.75, 5)
    lens_bins_centers = np.linspace(0.1, 0.6, 5)

    tracers: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    tracer_pairs: dict[
        tuple[TracerNames, str],
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], sacc.BandpowerWindow],
    ] = {}

    for i, mn in enumerate(src_bins_centers):
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.05 / 0.05)
        sacc_data.add_tracer("NZ", f"src{i}", z, dndz)
        tracers[f"src{i}"] = (z, dndz)

    for i, mn in enumerate(lens_bins_centers):
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.05 / 0.05)
        sacc_data.add_tracer("NZ", f"lens{i}", z, dndz)
        tracers[f"lens{i}"] = (z, dndz)

    for i, j in upper_triangle_indices(len(src_bins_centers)):

        weights = (
            np.eye(ells.shape[0], nobs, dtype=np.float64)
            + np.eye(ells.shape[0], nobs, k=5, dtype=np.float64)
            + np.eye(ells.shape[0], nobs, k=-5, dtype=np.float64)
        )
        window = sacc.BandpowerWindow(ells, weights)
        Cells = np.random.normal(size=nobs)
        sacc_data.add_ell_cl(
            "galaxy_shear_cl_ee",
            f"src{i}",
            f"src{j}",
            weights.T.dot(ells),
            Cells,
            window=window,
        )
        tracer_pairs[(TracerNames(f"src{i}", f"src{j}"), "galaxy_shear_cl_ee")] = (
            ells,
            Cells,
            window,
        )

    for i, j in upper_triangle_indices(len(lens_bins_centers)):
        weights = (
            np.eye(ells.shape[0], nobs, dtype=np.float64)
            + np.eye(ells.shape[0], nobs, k=5, dtype=np.float64)
            + np.eye(ells.shape[0], nobs, k=-5, dtype=np.float64)
        )
        window = sacc.BandpowerWindow(ells, weights)
        Cells = np.random.normal(size=nobs)
        sacc_data.add_ell_cl(
            "galaxy_density_cl",
            f"lens{i}",
            f"lens{j}",
            weights.T.dot(ells),
            Cells,
            window=window,
        )
        tracer_pairs[(TracerNames(f"lens{i}", f"lens{j}"), "galaxy_density_cl")] = (
            ells,
            Cells,
            window,
        )

    for i, j in product(range(len(src_bins_centers)), range(len(lens_bins_centers))):
        weights = (
            np.eye(ells.shape[0], nobs, dtype=np.float64)
            + np.eye(ells.shape[0], nobs, k=5, dtype=np.float64)
            + np.eye(ells.shape[0], nobs, k=-5, dtype=np.float64)
        )
        window = sacc.BandpowerWindow(ells, weights)
        Cells = np.random.normal(size=nobs)
        sacc_data.add_ell_cl(
            "galaxy_shearDensity_cl_e",
            f"src{i}",
            f"lens{j}",
            weights.T.dot(ells),
            Cells,
            window=window,
        )
        tracer_pairs[
            (TracerNames(f"src{i}", f"lens{j}"), "galaxy_shearDensity_cl_e")
        ] = (
            ells,
            Cells,
            window,
        )

    sacc_data.add_covariance(np.identity(len(sacc_data)) * 0.01)

    return sacc_data, tracers, tracer_pairs


@pytest.fixture(name="sacc_galaxy_xis")
def fixture_sacc_galaxy_xis():
    """Fixture for a SACC data without window functions."""

    z = np.linspace(0, 1.0, 256) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    sacc_data = sacc.Sacc()

    src_bins_centers = np.linspace(0.25, 0.75, 5)
    lens_bins_centers = np.linspace(0.1, 0.6, 5)

    tracers: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    tracer_pairs: dict[
        tuple[TracerNames, str], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
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
        tracer_pairs[(TracerNames(f"lens{i}", f"lens{j}"), "galaxy_density_xi")] = (
            thetas,
            xis,
        )
        dv.append(xis)

    for i, j in product(range(len(src_bins_centers)), range(len(lens_bins_centers))):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi(
            "galaxy_shearDensity_xi_t", f"src{i}", f"lens{j}", thetas, xis
        )
        tracer_pairs[
            (TracerNames(f"src{i}", f"lens{j}"), "galaxy_shearDensity_xi_t")
        ] = (
            thetas,
            xis,
        )
        dv.append(xis)

    for i, j in upper_triangle_indices(len(src_bins_centers)):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi(
            "galaxy_shear_xi_minus", f"src{i}", f"src{j}", thetas, xis
        )
        tracer_pairs[(TracerNames(f"src{i}", f"src{j}"), "galaxy_shear_xi_minus")] = (
            thetas,
            xis,
        )
        dv.append(xis)
    for i, j in upper_triangle_indices(len(src_bins_centers)):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi(
            "galaxy_shear_xi_plus", f"src{i}", f"src{j}", thetas, xis
        )
        tracer_pairs[(TracerNames(f"src{i}", f"src{j}"), "galaxy_shear_xi_plus")] = (
            thetas,
            xis,
        )
        dv.append(xis)

    delta_v = np.concatenate(dv, axis=0)
    cov = np.diag(np.ones_like(delta_v) * 0.01)

    sacc_data.add_covariance(cov)

    return sacc_data, tracers, tracer_pairs


@pytest.fixture(name="sacc_galaxy_xis_inverted")
def fixture_sacc_galaxy_xis_inverted():
    """Fixture for a SACC data without window functions."""

    z = np.linspace(0, 1.0, 256) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    sacc_data = sacc.Sacc()

    src_bins_centers = np.linspace(0.25, 0.75, 5)
    lens_bins_centers = np.linspace(0.1, 0.6, 5)

    tracers: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    tracer_pairs: dict[
        tuple[TracerNames, str], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
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
        sacc_data.add_theta_xi("galaxy_density_xi", f"lens{j}", f"lens{i}", thetas, xis)
        tracer_pairs[(TracerNames(f"lens{j}", f"lens{i}"), "galaxy_density_xi")] = (
            thetas,
            xis,
        )
        dv.append(xis)

    for i, j in product(range(len(src_bins_centers)), range(len(lens_bins_centers))):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi(
            "galaxy_shearDensity_xi_t", f"lens{i}", f"src{j}", thetas, xis
        )
        tracer_pairs[
            (TracerNames(f"lens{j}", f"src{i}"), "galaxy_shearDensity_xi_t")
        ] = (
            thetas,
            xis,
        )
        dv.append(xis)

    for i, j in upper_triangle_indices(len(src_bins_centers)):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi(
            "galaxy_shear_xi_minus", f"src{j}", f"src{i}", thetas, xis
        )
        tracer_pairs[(TracerNames(f"src{j}", f"src{i}"), "galaxy_shear_xi_minus")] = (
            thetas,
            xis,
        )
        dv.append(xis)
    for i, j in upper_triangle_indices(len(src_bins_centers)):
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi(
            "galaxy_shear_xi_plus", f"src{j}", f"src{i}", thetas, xis
        )
        tracer_pairs[(TracerNames(f"src{j}", f"src{i}"), "galaxy_shear_xi_plus")] = (
            thetas,
            xis,
        )
        dv.append(xis)

    delta_v = np.concatenate(dv, axis=0)
    cov = np.diag(np.ones_like(delta_v) * 0.01)

    sacc_data.add_covariance(cov)

    return sacc_data, tracers, tracer_pairs


@pytest.fixture(name="sacc_galaxy_cells_ambiguous")
def fixture_sacc_galaxy_cells_ambiguous() -> sacc.Sacc:
    """Fixture for a SACC data without window functions with ambiguous tracers."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "bin0", z, dndz)
    sacc_data.add_tracer("NZ", "bin1", z, dndz)
    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_shearDensity_cl_e", "bin0", "bin1", ells, Cells)
    cov = np.diag(np.zeros(len(ells)) + 0.01)

    sacc_data.add_covariance(cov)

    return sacc_data


@pytest.fixture(name="sacc_galaxy_cells_src0_src0_no_data")
def fixture_sacc_galaxy_cells_src0_src0_no_data():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz)

    return sacc_data, z, dndz


@pytest.fixture(name="sacc_galaxy_xis_lens0_lens0_no_data")
def fixture_sacc_galaxy_xis_lens0_lens0_no_data():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz)

    return sacc_data, z, thetas


# Two-point related SACC data fixtures with window functions


@pytest.fixture(name="sacc_galaxy_cells_src0_src0_window")
def fixture_sacc_galaxy_cells_src0_src0_window():
    """Fixture for a SACC data with a window function."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz)

    # Making a diagonal window function
    window = sacc.BandpowerWindow(ells, np.diag(np.ones_like(ells)))

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl(
        "galaxy_shear_cl_ee", "src0", "src0", ells, Cells, window=window
    )

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz


@pytest.fixture(name="sacc_galaxy_cells_src0_src0_no_window")
def fixture_sacc_galaxy_cells_src0_src0_no_window():
    """Fixture for a SACC data without a window function."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 256) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_shear_cl_ee", "src0", "src0", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz


@pytest.fixture(name="wl_factory")
def make_wl_factory():
    """Generate a WeakLensingFactory object."""
    return wl.WeakLensingFactory(per_bin_systematics=[], global_systematics=[])


@pytest.fixture(name="nc_factory")
def make_nc_factory():
    """Generate a NumberCountsFactory object."""
    return nc.NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
