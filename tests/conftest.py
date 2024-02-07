"""
Pytest configuration additions.

Fixtures defined here are available to any test in Firecrown.
"""

import pytest

import sacc

import numpy as np
import pyccl

from firecrown.likelihood.gauss_family.statistic.statistic import TrivialStatistic
from firecrown.parameters import ParamsMap
from firecrown.connector.mapping import MappingCosmoSIS, mapping_builder
from firecrown.modeling_tools import ModelingTools


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
