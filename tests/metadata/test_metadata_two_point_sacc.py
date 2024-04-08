"""Tests for the module firecrown.metadata.two_point.

In this module, we test the functions and classes involved SACC extraction tools.
"""

import re
import pytest
import numpy as np
from numpy.testing import assert_array_equal

import sacc

from firecrown.metadata.two_point import (
    extract_all_data_types_cells,
    extract_all_data_types_xi_thetas,
    extract_all_photoz_bin_combinations,
    extract_all_tracers,
    extract_window_function,
    GalaxyMeasuredType,
    TracerNames,
    TwoPointCells,
    TwoPointXiTheta,
)


@pytest.fixture(name="sacc_galaxy_src0_src0_invalid_data_type")
def fixture_sacc_galaxy_src0_src0_invalid_data_type():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz)

    Cells = np.random.normal(size=ells.shape[0])
    with pytest.warns(UserWarning):
        sacc_data.add_ell_cl("this_type_is_invalid", "src0", "src0", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz


@pytest.fixture(name="sacc_galaxy_xis_src0_lens0_bad_lens_label")
def fixture_sacc_galaxy_xis_src0_lens0_bad_lens_label():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "non_informative_label", z, dndz1)

    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi(
        "galaxy_shearDensity_xi_t", "src0", "non_informative_label", thetas, xis
    )

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_xis_src0_lens0_bad_source_label")
def fixture_sacc_galaxy_xis_src0_lens0_bad_source_label():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "non_informative_label", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz1)

    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi(
        "galaxy_shearDensity_xi_t", "non_informative_label", "lens0", thetas, xis
    )

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_xis_inconsistent_lens_label")
def fixture_sacc_galaxy_xis_inconsistent_lens_label():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz1)

    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi(
        "cmbGalaxy_convergenceShear_xi_t", "src0", "lens0", thetas, xis
    )

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_xis_inconsistent_source_label")
def fixture_sacc_galaxy_xis_inconsistent_source_label():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz1)

    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi("clusterGalaxy_density_xi", "src0", "lens0", thetas, xis)

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1


@pytest.fixture(name="sacc_galaxy_xis_three_tracers")
def fixture_sacc_galaxy_xis_three_tracers():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz1)

    dndz2 = np.exp(-0.5 * (z - 0.9) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens1", z, dndz2)

    xis = np.random.normal(size=thetas.shape[0])
    for theta, xi in zip(thetas, xis):
        sacc_data.add_data_point(
            "galaxy_density_xi", ("src0", "lens0", "lens1"), xi, theta=theta
        )

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1, dndz2


@pytest.fixture(name="sacc_galaxy_cells_three_tracers")
def fixture_sacc_galaxy_cells_three_tracers():
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz1)

    dndz2 = np.exp(-0.5 * (z - 0.9) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens1", z, dndz2)

    Cells = np.random.normal(size=ells.shape[0])
    for ell, cell in zip(ells, Cells):
        sacc_data.add_data_point(
            "galaxy_shear_cl_ee", ("src0", "lens0", "lens1"), cell, ell=ell
        )

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1, dndz2


def test_extract_all_tracers_cells(sacc_galaxy_cells):
    sacc_data, tracers, _ = sacc_galaxy_cells
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    for tracer in all_tracers:
        orig_tracer = tracers[tracer.bin_name]
        assert_array_equal(tracer.z, orig_tracer[0])
        assert_array_equal(tracer.dndz, orig_tracer[1])


def test_extract_all_tracers_xis(sacc_galaxy_xis):
    sacc_data, tracers, _ = sacc_galaxy_xis
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    for tracer in all_tracers:
        orig_tracer = tracers[tracer.bin_name]
        assert_array_equal(tracer.z, orig_tracer[0])
        assert_array_equal(tracer.dndz, orig_tracer[1])


def test_extract_all_tracers_cells_src0_src0(sacc_galaxy_cells_src0_src0):
    sacc_data, z, dndz = sacc_galaxy_cells_src0_src0
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 1

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert_array_equal(tracer.dndz, dndz)
        assert tracer.bin_name == "src0"
        assert tracer.measured_type == GalaxyMeasuredType.SHEAR_E


def test_extract_all_tracers_cells_src0_src1(sacc_galaxy_cells_src0_src1):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_cells_src0_src1
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert tracer.measured_type == GalaxyMeasuredType.SHEAR_E
        if tracer.bin_name == "src0":
            assert_array_equal(tracer.dndz, dndz0)
        elif tracer.bin_name == "src1":
            assert_array_equal(tracer.dndz, dndz1)


def test_extract_all_tracers_cells_lens0_lens0(sacc_galaxy_cells_lens0_lens0):
    sacc_data, z, dndz = sacc_galaxy_cells_lens0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 1

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert_array_equal(tracer.dndz, dndz)
        assert tracer.bin_name == "lens0"
        assert tracer.measured_type == GalaxyMeasuredType.COUNTS


def test_extract_all_tracers_cells_lens0_lens1(sacc_galaxy_cells_lens0_lens1):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_cells_lens0_lens1
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert tracer.measured_type == GalaxyMeasuredType.COUNTS
        if tracer.bin_name == "lens0":
            assert_array_equal(tracer.dndz, dndz0)
        elif tracer.bin_name == "lens1":
            assert_array_equal(tracer.dndz, dndz1)


def test_extract_all_tracers_xis_lens0_lens0(sacc_galaxy_xis_lens0_lens0):
    sacc_data, z, dndz = sacc_galaxy_xis_lens0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 1

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert_array_equal(tracer.dndz, dndz)
        assert tracer.bin_name == "lens0"
        assert tracer.measured_type == GalaxyMeasuredType.COUNTS


def test_extract_all_tracers_xis_lens0_lens1(sacc_galaxy_xis_lens0_lens1):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_xis_lens0_lens1
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert tracer.measured_type == GalaxyMeasuredType.COUNTS
        if tracer.bin_name == "lens0":
            assert_array_equal(tracer.dndz, dndz0)
        elif tracer.bin_name == "lens1":
            assert_array_equal(tracer.dndz, dndz1)


def test_extract_all_trace_cells_src0_lens0(sacc_galaxy_cells_src0_lens0):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_cells_src0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        if tracer.bin_name == "src0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz0)
            assert tracer.measured_type == GalaxyMeasuredType.SHEAR_E
        elif tracer.bin_name == "lens0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz1)
            assert tracer.measured_type == GalaxyMeasuredType.COUNTS


def test_extract_all_trace_xis_src0_lens0(sacc_galaxy_xis_src0_lens0):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_xis_src0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        if tracer.bin_name == "src0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz0)
            assert tracer.measured_type == GalaxyMeasuredType.SHEAR_T
        elif tracer.bin_name == "lens0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz1)
            assert tracer.measured_type == GalaxyMeasuredType.COUNTS


def test_extract_all_tracers_invalid_data_type(
    sacc_galaxy_src0_src0_invalid_data_type,
):
    sacc_data, _, _ = sacc_galaxy_src0_src0_invalid_data_type
    assert sacc_data is not None
    with pytest.raises(
        ValueError, match="Tracer src0 does not have data points associated with it."
    ):
        _ = extract_all_tracers(sacc_data)


def test_extract_all_tracers_bad_lens_label(
    sacc_galaxy_xis_src0_lens0_bad_lens_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_src0_lens0_bad_lens_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match="Tracer non_informative_label does not have a compatible measured type.",
    ):
        _ = extract_all_tracers(sacc_data)


def test_extract_all_tracers_bad_source_label(
    sacc_galaxy_xis_src0_lens0_bad_source_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_src0_lens0_bad_source_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match=(
            "Tracer non_informative_label does not have a compatible measured type."
        ),
    ):
        _ = extract_all_tracers(sacc_data)


def test_extract_all_tracers_inconsistent_lens_label(
    sacc_galaxy_xis_inconsistent_lens_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_inconsistent_lens_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match=(
            "Tracer lens0 matches the lens regex but does "
            "not have a compatible measured type."
        ),
    ):
        _ = extract_all_tracers(sacc_data)


def test_extract_all_tracers_inconsistent_source_label(
    sacc_galaxy_xis_inconsistent_source_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_inconsistent_source_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match=(
            "Tracer src0 matches the source regex but does "
            "not have a compatible measured type."
        ),
    ):
        _ = extract_all_tracers(sacc_data)


def test_extract_all_data_cells(sacc_galaxy_cells):
    sacc_data, _, tracer_pairs = sacc_galaxy_cells

    all_data = extract_all_data_types_cells(sacc_data)
    assert len(all_data) == len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert tracer_names in tracer_pairs

        tracer_pair = tracer_pairs[tracer_names]
        assert_array_equal(two_point["ells"], tracer_pair[1])
        assert two_point["data_type"] == tracer_pair[0]


def test_extract_all_data_cells_by_type(sacc_galaxy_cells):
    sacc_data, _, tracer_pairs = sacc_galaxy_cells

    tracer_pairs = {
        k: v for k, v in tracer_pairs.items() if v[0] == "galaxy_shear_cl_ee"
    }
    all_data = extract_all_data_types_cells(
        sacc_data, allowed_data_type=["galaxy_shear_cl_ee"]
    )
    assert len(all_data) == len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert tracer_names in tracer_pairs

        tracer_pair = tracer_pairs[tracer_names]
        assert_array_equal(two_point["ells"], tracer_pair[1])
        assert two_point["data_type"] == tracer_pair[0]


def test_extract_all_data_xis(sacc_galaxy_xis):
    sacc_data, _, tracer_pairs = sacc_galaxy_xis

    all_data = extract_all_data_types_xi_thetas(sacc_data)
    assert len(all_data) == len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert tracer_names in tracer_pairs

        tracer_pair = tracer_pairs[tracer_names]
        assert_array_equal(two_point["thetas"], tracer_pair[1])
        assert two_point["data_type"] == tracer_pair[0]


def test_extract_all_data_xis_by_type(sacc_galaxy_xis):
    sacc_data, _, tracer_pairs = sacc_galaxy_xis

    tracer_pairs = {
        k: v for k, v in tracer_pairs.items() if v[0] == "galaxy_density_xi"
    }

    all_data = extract_all_data_types_xi_thetas(
        sacc_data, allowed_data_type=["galaxy_density_xi"]
    )
    assert len(all_data) == len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert tracer_names in tracer_pairs

        tracer_pair = tracer_pairs[tracer_names]
        assert_array_equal(two_point["thetas"], tracer_pair[1])
        assert two_point["data_type"] == tracer_pair[0]


def test_extract_no_window(sacc_galaxy_cells_src0_src0):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0
    indices = np.array([0, 1, 2], dtype=np.int64)

    with pytest.warns(UserWarning):
        assert extract_window_function(sacc_data, indices=indices) is None


def test_extract_all_data_types_three_tracers_xis(sacc_galaxy_xis_three_tracers):
    sacc_data, _, _, _, _ = sacc_galaxy_xis_three_tracers

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Tracer combination ('src0', 'lens0', 'lens1') "
            "does not have exactly two tracers."
        ),
    ):
        _ = extract_all_data_types_xi_thetas(sacc_data)


def test_extract_all_data_types_three_tracers_cells(sacc_galaxy_cells_three_tracers):
    sacc_data, _, _, _, _ = sacc_galaxy_cells_three_tracers

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Tracer combination ('src0', 'lens0', 'lens1') "
            "does not have exactly two tracers."
        ),
    ):
        _ = extract_all_data_types_cells(sacc_data)


def test_extract_all_photoz_bin_combinations_xis(sacc_galaxy_xis):
    sacc_data, _, tracer_pairs = sacc_galaxy_xis
    # We build all possible combinations of tracers
    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    # We get all combinations already present in the data
    tracer_names_list = [
        TracerNames(bin_comb.x.bin_name, bin_comb.y.bin_name)
        for bin_comb in all_bin_combs
    ]

    # We check if the particular are present in the list
    for tracer_names in tracer_pairs:
        assert tracer_names in tracer_names_list

        bin_comb = all_bin_combs[tracer_names_list.index(tracer_names)]
        two_point_xis = TwoPointXiTheta(
            XY=bin_comb, thetas=np.linspace(0.0, 2.0 * np.pi, 20)
        )
        assert two_point_xis.get_sacc_name() == tracer_pairs[tracer_names][0]


def test_extract_all_photoz_bin_combinations_cells(sacc_galaxy_cells):
    sacc_data, _, tracer_pairs = sacc_galaxy_cells
    # We build all possible combinations of tracers
    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    # We get all combinations already present in the data
    tracer_names_list = [
        TracerNames(bin_comb.x.bin_name, bin_comb.y.bin_name)
        for bin_comb in all_bin_combs
    ]

    # We check if the particular are present in the list
    for tracer_names in tracer_pairs:
        assert tracer_names in tracer_names_list

        bin_comb = all_bin_combs[tracer_names_list.index(tracer_names)]
        two_point_cells = TwoPointCells(
            XY=bin_comb, ells=np.unique(np.logspace(1, 3, 10))
        )
        assert two_point_cells.get_sacc_name() == tracer_pairs[tracer_names][0]


def test_make_cells(sacc_galaxy_cells):
    sacc_data, _, _ = sacc_galaxy_cells
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    for bin_comb in all_bin_combs:
        two_point_cells = TwoPointCells(XY=bin_comb, ells=ells)

        assert two_point_cells.ells is not None
        assert_array_equal(two_point_cells.ells, ells)
        assert two_point_cells.XY is not None
        assert two_point_cells.XY == bin_comb


def test_make_xis(sacc_galaxy_xis):
    sacc_data, _, _ = sacc_galaxy_xis
    thetas = np.linspace(0.0, 2.0 * np.pi, 20)

    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    for bin_comb in all_bin_combs:
        two_point_xis = TwoPointXiTheta(XY=bin_comb, thetas=thetas)

        assert two_point_xis.thetas is not None
        assert_array_equal(two_point_xis.thetas, thetas)
        assert two_point_xis.XY is not None
        assert two_point_xis.XY == bin_comb
