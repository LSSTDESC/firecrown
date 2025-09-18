"""Tests for the modules firecrown.metata_types and firecrown.metadata_functions.

In this module, we test the functions and classes involved SACC extraction tools.
"""

import re
import pytest
import numpy as np
from numpy.testing import assert_array_equal

import sacc

from firecrown.parameters import ParamsMap
from firecrown.metadata_types import (
    Galaxies,
    TracerNames,
    TwoPointHarmonic,
    TwoPointReal,
    InferredGalaxyZDist,
    CMB,
    TypeSource,
    type_to_sacc_string_harmonic,
    type_to_sacc_string_real,
)
from firecrown.metadata_functions import (
    extract_all_harmonic_metadata_indices,
    extract_all_harmonic_metadata,
    extract_all_photoz_bin_combinations,
    extract_all_real_metadata_indices,
    extract_all_real_metadata,
    extract_all_tracers_inferred_galaxy_zdists,
    extract_window_function,
    make_all_photoz_bin_combinations_with_cmb,
    make_all_photoz_bin_combinations,
)
from firecrown.data_functions import (
    check_two_point_consistence_harmonic,
    check_two_point_consistence_real,
    extract_all_harmonic_data,
    extract_all_real_data,
)
from firecrown.likelihood.two_point import TwoPoint, TwoPointFactory, use_source_factory


@pytest.fixture(name="sacc_galaxy_src0_src0_invalid_data_type")
def fixture_sacc_galaxy_src0_src0_invalid_data_type(
    recwarn,
) -> tuple[sacc.Sacc, np.ndarray, np.ndarray]:
    """Fixture for a SACC data without window functions."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("this_type_is_invalid", "src0", "src0", ells, Cells)
    warning = next(iter(recwarn), None)
    if warning is not None:
        assert isinstance(warning.message, UserWarning)
        assert re.match(
            r"Unknown data_type value this_type_is_invalid\.", str(warning.message)
        )

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz


@pytest.fixture(name="sacc_galaxy_xis_src0_lens0_bad_lens_label")
def fixture_sacc_galaxy_xis_src0_lens0_bad_lens_label() -> (
    tuple[sacc.Sacc, np.ndarray, np.ndarray, np.ndarray]
):
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
def fixture_sacc_galaxy_xis_src0_lens0_bad_source_label() -> (
    tuple[sacc.Sacc, np.ndarray, np.ndarray, np.ndarray]
):
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
def fixture_sacc_galaxy_xis_inconsistent_lens_label() -> (
    tuple[sacc.Sacc, np.ndarray, np.ndarray, np.ndarray]
):
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
def fixture_sacc_galaxy_xis_inconsistent_source_label() -> (
    tuple[sacc.Sacc, np.ndarray, np.ndarray, np.ndarray]
):
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
def fixture_sacc_galaxy_xis_three_tracers() -> (
    tuple[sacc.Sacc, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
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

    # We have to coerce the return type of np.random.normal to np.array,
    # because mypy 1.16 is unable to infer the type of the return value.
    xis = np.array(np.random.normal(size=thetas.shape[0]))
    for theta, xi in zip(thetas, xis):
        sacc_data.add_data_point(
            "galaxy_density_xi", ("src0", "lens0", "lens1"), xi, theta=theta
        )

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1, dndz2


@pytest.fixture(name="sacc_galaxy_cells_three_tracers")
def fixture_sacc_galaxy_cells_three_tracers() -> (
    tuple[sacc.Sacc, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
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

    # We have to coerce the return type of np.random.normal to np.array,
    # because mypy 1.16 is unable to infer the type of the return value.
    Cells = np.array(np.random.normal(size=ells.shape[0]))
    for ell, cell in zip(ells, Cells):
        sacc_data.add_data_point(
            "galaxy_shear_cl_ee", ("src0", "lens0", "lens1"), cell, ell=ell
        )

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    return sacc_data, z, dndz0, dndz1, dndz2


def test_extract_all_tracers_cells(sacc_galaxy_cells) -> None:
    sacc_data, tracers, _ = sacc_galaxy_cells
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    for tracer in all_tracers:
        orig_tracer = tracers[tracer.bin_name]
        assert_array_equal(tracer.z, orig_tracer[0])
        assert_array_equal(tracer.dndz, orig_tracer[1])


def test_extract_all_tracers_xis(sacc_galaxy_xis):
    sacc_data, tracers, _ = sacc_galaxy_xis
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    for tracer in all_tracers:
        orig_tracer = tracers[tracer.bin_name]
        assert_array_equal(tracer.z, orig_tracer[0])
        assert_array_equal(tracer.dndz, orig_tracer[1])


def test_extract_all_tracers_cells_src0_src0(sacc_galaxy_cells_src0_src0):
    sacc_data, z, dndz = sacc_galaxy_cells_src0_src0
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 1

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert_array_equal(tracer.dndz, dndz)
        assert tracer.bin_name == "src0"
        assert tracer.measurements == {Galaxies.SHEAR_E}


def test_extract_all_tracers_cells_src0_src1(sacc_galaxy_cells_src0_src1):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_cells_src0_src1
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert tracer.measurements == {Galaxies.SHEAR_E}
        if tracer.bin_name == "src0":
            assert_array_equal(tracer.dndz, dndz0)
        elif tracer.bin_name == "src1":
            assert_array_equal(tracer.dndz, dndz1)


def test_extract_all_tracers_cells_lens0_lens0(sacc_galaxy_cells_lens0_lens0):
    sacc_data, z, dndz = sacc_galaxy_cells_lens0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 1

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert_array_equal(tracer.dndz, dndz)
        assert tracer.bin_name == "lens0"
        assert tracer.measurements == {Galaxies.COUNTS}


def test_extract_all_tracers_cells_lens0_lens1(sacc_galaxy_cells_lens0_lens1):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_cells_lens0_lens1
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert tracer.measurements == {Galaxies.COUNTS}
        if tracer.bin_name == "lens0":
            assert_array_equal(tracer.dndz, dndz0)
        elif tracer.bin_name == "lens1":
            assert_array_equal(tracer.dndz, dndz1)


def test_extract_all_tracers_xis_lens0_lens0(sacc_galaxy_xis_lens0_lens0):
    sacc_data, z, dndz = sacc_galaxy_xis_lens0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 1

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert_array_equal(tracer.dndz, dndz)
        assert tracer.bin_name == "lens0"
        assert tracer.measurements == {Galaxies.COUNTS}


def test_extract_all_tracers_xis_lens0_lens1(sacc_galaxy_xis_lens0_lens1):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_xis_lens0_lens1
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        assert_array_equal(tracer.z, z)
        assert tracer.measurements == {Galaxies.COUNTS}
        if tracer.bin_name == "lens0":
            assert_array_equal(tracer.dndz, dndz0)
        elif tracer.bin_name == "lens1":
            assert_array_equal(tracer.dndz, dndz1)


def test_extract_all_trace_cells_src0_lens0(sacc_galaxy_cells_src0_lens0):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_cells_src0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        if tracer.bin_name == "src0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz0)
            assert tracer.measurements == {Galaxies.SHEAR_E}
        elif tracer.bin_name == "lens0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz1)
            assert tracer.measurements == {Galaxies.COUNTS}


def test_extract_all_trace_xis_src0_lens0(sacc_galaxy_xis_src0_lens0):
    sacc_data, z, dndz0, dndz1 = sacc_galaxy_xis_src0_lens0
    assert sacc_data is not None
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    assert len(all_tracers) == 2

    for tracer in all_tracers:
        if tracer.bin_name == "src0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz0)
            assert tracer.measurements == {Galaxies.SHEAR_T}
        elif tracer.bin_name == "lens0":
            assert_array_equal(tracer.z, z)
            assert_array_equal(tracer.dndz, dndz1)
            assert tracer.measurements == {Galaxies.COUNTS}


def test_extract_all_tracers_invalid_data_type(
    sacc_galaxy_src0_src0_invalid_data_type,
):
    sacc_data, _, _ = sacc_galaxy_src0_src0_invalid_data_type
    assert sacc_data is not None
    with pytest.raises(
        ValueError, match="Tracer src0 does not have data points associated with it."
    ):
        _ = extract_all_tracers_inferred_galaxy_zdists(sacc_data)


def test_extract_all_tracers_bad_lens_label(
    sacc_galaxy_xis_src0_lens0_bad_lens_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_src0_lens0_bad_lens_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match="Tracer src0 does not have data points associated with it.",
    ):
        _ = extract_all_tracers_inferred_galaxy_zdists(sacc_data)


def test_extract_all_tracers_bad_source_label(
    sacc_galaxy_xis_src0_lens0_bad_source_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_src0_lens0_bad_source_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match=(
            "Tracer non_informative_label does not have data points associated with it."
        ),
    ):
        _ = extract_all_tracers_inferred_galaxy_zdists(sacc_data)


def test_extract_all_tracers_inconsistent_lens_label(
    sacc_galaxy_xis_inconsistent_lens_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_inconsistent_lens_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match=("Invalid SACC file, tracer names do not respect the naming convetion."),
    ):
        _ = extract_all_tracers_inferred_galaxy_zdists(sacc_data)


def test_extract_all_tracers_inconsistent_source_label(
    sacc_galaxy_xis_inconsistent_source_label,
):
    sacc_data, _, _, _ = sacc_galaxy_xis_inconsistent_source_label
    assert sacc_data is not None
    with pytest.raises(
        ValueError,
        match=("Invalid SACC file, tracer names do not respect the naming convetion."),
    ):
        _ = extract_all_tracers_inferred_galaxy_zdists(sacc_data)


def test_extract_all_metadata_index_harmonics(sacc_galaxy_cells):
    sacc_data, _, tracer_pairs = sacc_galaxy_cells

    all_data = extract_all_harmonic_metadata_indices(sacc_data)
    assert len(all_data) == len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert (tracer_names, two_point["data_type"]) in tracer_pairs


def test_extract_all_metadata_index_harmonics_by_type(sacc_galaxy_cells):
    sacc_data, _, tracer_pairs = sacc_galaxy_cells

    all_data = extract_all_harmonic_metadata_indices(
        sacc_data, allowed_data_type=["galaxy_shear_cl_ee"]
    )
    assert len(all_data) < len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert (tracer_names, "galaxy_shear_cl_ee") in tracer_pairs


def test_extract_all_metadata_index_reals(sacc_galaxy_xis):
    sacc_data, _, tracer_pairs = sacc_galaxy_xis

    all_data = extract_all_real_metadata_indices(sacc_data)
    assert len(all_data) == len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert (tracer_names, two_point["data_type"]) in tracer_pairs


def test_extract_all_metadata_index_reals_by_type(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict],
) -> None:
    sacc_data, _, tracer_pairs = sacc_galaxy_xis

    all_data = extract_all_real_metadata_indices(
        sacc_data, allowed_data_type=["galaxy_density_xi"]
    )
    assert len(all_data) < len(tracer_pairs)

    for two_point in all_data:
        tracer_names = two_point["tracer_names"]
        assert (tracer_names, two_point["data_type"]) in tracer_pairs


def test_extract_no_window(
    sacc_galaxy_cells_src0_src0: tuple[sacc.Sacc, dict, dict],
) -> None:
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0
    indices = np.array([0, 1, 2], dtype=np.int64)

    with pytest.warns(UserWarning):
        window = extract_window_function(sacc_data, indices=indices)
        assert window[0] is None
        assert window[1] is None


def test_extract_all_data_types_three_tracers_reals(
    sacc_galaxy_xis_three_tracers: tuple[sacc.Sacc, dict, dict, dict, dict],
) -> None:
    sacc_data, _, _, _, _ = sacc_galaxy_xis_three_tracers

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Tracer combination ('src0', 'lens0', 'lens1') "
            "does not have exactly two tracers."
        ),
    ):
        _ = extract_all_real_metadata_indices(sacc_data)


def test_extract_all_data_types_three_tracers_harmonics(
    sacc_galaxy_cells_three_tracers: tuple[sacc.Sacc, dict, dict, dict, dict],
) -> None:
    sacc_data, _, _, _, _ = sacc_galaxy_cells_three_tracers

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Tracer combination ('src0', 'lens0', 'lens1') "
            "does not have exactly two tracers."
        ),
    ):
        _ = extract_all_harmonic_metadata_indices(sacc_data)


def test_extract_all_photoz_bin_combinations_reals(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict],
) -> None:
    sacc_data, _, tracer_pairs = sacc_galaxy_xis
    # We build all possible combinations of tracers
    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    # We get all combinations already present in the data
    tracer_names_list = [
        (
            TracerNames(bin_comb.x.bin_name, bin_comb.y.bin_name),
            type_to_sacc_string_real(bin_comb.x_measurement, bin_comb.y_measurement),
        )
        for bin_comb in all_bin_combs
    ]

    # We check if the particular are present in the list
    for tracer_names_type in tracer_pairs:
        assert tracer_names_type in tracer_names_list

        bin_comb = all_bin_combs[tracer_names_list.index(tracer_names_type)]
        two_point_xis = TwoPointReal(
            XY=bin_comb, thetas=np.linspace(0.0, 2.0 * np.pi, 20, dtype=np.float64)
        )
        assert two_point_xis.get_sacc_name() == tracer_names_type[1]


def test_extract_all_photoz_bin_combinations_harmonics(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
) -> None:
    sacc_data, _, tracer_pairs = sacc_galaxy_cells
    # We build all possible combinations of tracers
    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    # We get all combinations already present in the data
    tracer_names_list = [
        (
            TracerNames(bin_comb.x.bin_name, bin_comb.y.bin_name),
            type_to_sacc_string_harmonic(
                bin_comb.x_measurement, bin_comb.y_measurement
            ),
        )
        for bin_comb in all_bin_combs
    ]

    # We check if the particular are present in the list
    for tracer_names_type in tracer_pairs:
        assert tracer_names_type in tracer_names_list

        bin_comb = all_bin_combs[tracer_names_list.index(tracer_names_type)]
        two_point_cells = TwoPointHarmonic(
            XY=bin_comb, ells=np.unique(np.logspace(1, 3, 10))
        )
        assert two_point_cells.get_sacc_name() == tracer_names_type[1]


def test_make_harmonics(sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict]) -> None:
    sacc_data, _, _ = sacc_galaxy_cells
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    for bin_comb in all_bin_combs:
        two_point_cells = TwoPointHarmonic(XY=bin_comb, ells=ells)

        assert two_point_cells.ells is not None
        assert_array_equal(two_point_cells.ells, ells)
        assert two_point_cells.XY is not None
        assert two_point_cells.XY == bin_comb


def test_make_reals(sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict]) -> None:
    sacc_data, _, _ = sacc_galaxy_xis
    thetas = np.linspace(0.0, 2.0 * np.pi, 20, dtype=np.float64)

    all_bin_combs = extract_all_photoz_bin_combinations(sacc_data)

    for bin_comb in all_bin_combs:
        two_point_xis = TwoPointReal(XY=bin_comb, thetas=thetas)

        assert two_point_xis.thetas is not None
        assert_array_equal(two_point_xis.thetas, thetas)
        assert two_point_xis.XY is not None
        assert two_point_xis.XY == bin_comb


def test_extract_all_harmonic_data(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
) -> None:
    sacc_data, _, tracer_pairs = sacc_galaxy_cells

    two_point_harmonics = extract_all_harmonic_data(sacc_data)
    assert len(two_point_harmonics) == len(tracer_pairs)

    for two_point in two_point_harmonics:
        metadata = two_point.metadata
        assert isinstance(metadata, TwoPointHarmonic)
        tracer_names = TracerNames(metadata.XY.x.bin_name, metadata.XY.y.bin_name)
        assert (tracer_names, metadata.get_sacc_name()) in tracer_pairs

        ells, Cell = tracer_pairs[(tracer_names, metadata.get_sacc_name())]
        assert_array_equal(metadata.ells, ells)
        assert_array_equal(two_point.data, Cell)

    check_two_point_consistence_harmonic(two_point_harmonics)


def test_extract_all_harmonic_with_window_data(
    sacc_galaxy_cwindows: tuple[sacc.Sacc, dict, dict],
) -> None:
    sacc_data, _, tracer_pairs = sacc_galaxy_cwindows

    two_point_harmonics = extract_all_harmonic_data(sacc_data)
    assert len(two_point_harmonics) == len(tracer_pairs)

    for two_point in two_point_harmonics:
        metadata = two_point.metadata
        assert isinstance(metadata, TwoPointHarmonic)
        tracer_names = TracerNames(metadata.XY.x.bin_name, metadata.XY.y.bin_name)
        assert (tracer_names, metadata.get_sacc_name()) in tracer_pairs

        ells, Cell, window = tracer_pairs[(tracer_names, metadata.get_sacc_name())]
        assert_array_equal(metadata.ells, ells)
        assert_array_equal(two_point.data, Cell)
        assert metadata.window is not None

        assert_array_equal(metadata.window, window.weight / window.weight.sum(axis=0))
        assert_array_equal(metadata.ells, window.values)

    check_two_point_consistence_harmonic(two_point_harmonics)


def test_extract_all_real_data(sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict]):
    sacc_data, _, tracer_pairs = sacc_galaxy_xis

    two_point_xis = extract_all_real_data(sacc_data)
    assert len(two_point_xis) == len(tracer_pairs)

    for two_point in two_point_xis:
        metadata = two_point.metadata
        assert isinstance(metadata, TwoPointReal)
        tracer_names = TracerNames(metadata.XY.x.bin_name, metadata.XY.y.bin_name)
        assert (tracer_names, metadata.get_sacc_name()) in tracer_pairs

        thetas, xi = tracer_pairs[(tracer_names, metadata.get_sacc_name())]
        assert_array_equal(metadata.thetas, thetas)
        assert_array_equal(two_point.data, xi)

    check_two_point_consistence_real(two_point_xis)


def test_constructor_harmonic_metadata(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict], tp_factory: TwoPointFactory
):
    sacc_data, _, tracer_pairs = sacc_galaxy_cells

    two_point_harmonics = extract_all_harmonic_metadata(sacc_data)

    two_points_new = TwoPoint.from_metadata(two_point_harmonics, tp_factory)

    assert two_points_new is not None

    for two_point in two_points_new:
        tracer_pairs_key = (two_point.sacc_tracers, two_point.sacc_data_type)
        assert tracer_pairs_key in tracer_pairs
        assert two_point.ells is not None
        assert_array_equal(two_point.ells, tracer_pairs[tracer_pairs_key][0])


def test_constructor_harmonic_data(sacc_galaxy_cells, tp_factory):
    sacc_data, _, tracer_pairs = sacc_galaxy_cells

    two_point_harmonics = extract_all_harmonic_data(sacc_data)

    check_two_point_consistence_harmonic(two_point_harmonics)
    two_points_new = TwoPoint.from_measurement(two_point_harmonics, tp_factory)

    assert two_points_new is not None

    for two_point in two_points_new:
        tracer_pairs_key = (two_point.sacc_tracers, two_point.sacc_data_type)
        assert tracer_pairs_key in tracer_pairs
        assert two_point.ells is not None
        assert_array_equal(two_point.ells, tracer_pairs[tracer_pairs_key][0])
        assert_array_equal(
            two_point.get_data_vector(), tracer_pairs[tracer_pairs_key][1]
        )


def test_constructor_harmonic_with_window_metadata(sacc_galaxy_cwindows, tp_factory):
    sacc_data, _, tracer_pairs = sacc_galaxy_cwindows

    two_point_harmonics = extract_all_harmonic_metadata(sacc_data)

    two_points_new = TwoPoint.from_metadata(two_point_harmonics, tp_factory)

    assert two_points_new is not None

    for two_point in two_points_new:
        tracer_pairs_key = (two_point.sacc_tracers, two_point.sacc_data_type)
        _, _, window = tracer_pairs[tracer_pairs_key]

        assert tracer_pairs_key in tracer_pairs
        assert two_point.ells is not None
        assert two_point.window is not None
        assert_array_equal(
            two_point.window,
            window.weight / window.weight.sum(axis=0),
        )
        assert_array_equal(two_point.ells, window.values)


def test_constructor_harmonic_with_window_data(sacc_galaxy_cwindows, tp_factory):
    sacc_data, _, tracer_pairs = sacc_galaxy_cwindows

    two_point_harmonics = extract_all_harmonic_data(sacc_data)

    check_two_point_consistence_harmonic(two_point_harmonics)
    two_points_new = TwoPoint.from_measurement(two_point_harmonics, tp_factory)

    assert two_points_new is not None

    for two_point in two_points_new:
        tracer_pairs_key = (two_point.sacc_tracers, two_point.sacc_data_type)
        _, Cell, window = tracer_pairs[tracer_pairs_key]

        assert tracer_pairs_key in tracer_pairs
        assert two_point.ells is not None
        assert_array_equal(two_point.get_data_vector(), Cell)
        assert two_point.window is not None
        assert_array_equal(
            two_point.window,
            window.weight / window.weight.sum(axis=0),
        )
        assert_array_equal(two_point.ells, window.values)


def test_constructor_reals_metadata(sacc_galaxy_xis, tp_factory):
    sacc_data, _, tracer_pairs = sacc_galaxy_xis

    two_point_reals = extract_all_real_metadata(sacc_data)

    two_points_new = TwoPoint.from_metadata(two_point_reals, tp_factory)

    assert two_points_new is not None

    for two_point in two_points_new:
        tracer_pairs_key = (two_point.sacc_tracers, two_point.sacc_data_type)
        assert tracer_pairs_key in tracer_pairs
        assert two_point.thetas is not None
        assert_array_equal(two_point.thetas, tracer_pairs[tracer_pairs_key][0])


def test_constructor_reals_data(sacc_galaxy_xis, tp_factory):
    sacc_data, _, tracer_pairs = sacc_galaxy_xis

    two_point_reals = extract_all_real_data(sacc_data)

    check_two_point_consistence_real(two_point_reals)
    two_points_new = TwoPoint.from_measurement(two_point_reals, tp_factory)

    assert two_points_new is not None

    for two_point in two_points_new:
        tracer_pairs_key = (two_point.sacc_tracers, two_point.sacc_data_type)
        assert tracer_pairs_key in tracer_pairs
        assert two_point.thetas is not None
        assert_array_equal(two_point.thetas, tracer_pairs[tracer_pairs_key][0])
        assert_array_equal(
            two_point.get_data_vector(), tracer_pairs[tracer_pairs_key][1]
        )


def test_compare_constructors_harmonic(
    sacc_galaxy_cells, tp_factory, tools_with_vanilla_cosmology
):
    sacc_data, _, _ = sacc_galaxy_cells

    two_point_harmonics = extract_all_harmonic_data(sacc_data)

    two_points_harmonic = TwoPoint.from_measurement(two_point_harmonics, tp_factory)

    params = ParamsMap(two_points_harmonic.required_parameters().get_default_values())

    two_points_old = []
    for tpm in two_point_harmonics:
        sacc_data_type = tpm.metadata.get_sacc_name()
        source0 = use_source_factory(
            tpm.metadata.XY.x, tpm.metadata.XY.x_measurement, tp_factory=tp_factory
        )
        source1 = use_source_factory(
            tpm.metadata.XY.y, tpm.metadata.XY.y_measurement, tp_factory=tp_factory
        )
        two_point = TwoPoint(sacc_data_type, source0, source1)
        two_point.read(sacc_data)
        two_points_old.append(two_point)

    assert len(two_points_harmonic) == len(two_points_old)

    for two_point, two_point_old in zip(two_points_harmonic, two_points_old):
        assert two_point.sacc_tracers == two_point_old.sacc_tracers
        assert two_point.sacc_data_type == two_point_old.sacc_data_type
        assert two_point.ells is not None
        assert two_point_old.ells is not None
        assert_array_equal(two_point.ells, two_point_old.ells)
        assert_array_equal(two_point.get_data_vector(), two_point_old.get_data_vector())
        assert two_point.sacc_indices is not None
        assert two_point_old.sacc_indices is not None
        assert_array_equal(two_point.sacc_indices, two_point_old.sacc_indices)

        two_point.update(params)
        two_point_old.update(params)

        assert_array_equal(
            two_point.compute_theory_vector(tools_with_vanilla_cosmology),
            two_point_old.compute_theory_vector(tools_with_vanilla_cosmology),
        )


def test_compare_constructors_harmonic_with_window(
    sacc_galaxy_cwindows, tp_factory, tools_with_vanilla_cosmology
):
    sacc_data, _, _ = sacc_galaxy_cwindows

    two_point_harmonics = extract_all_harmonic_data(sacc_data)

    two_points_harmonic = TwoPoint.from_measurement(two_point_harmonics, tp_factory)

    params = ParamsMap(two_points_harmonic.required_parameters().get_default_values())

    two_points_old = []
    for tpm in two_point_harmonics:
        sacc_data_type = tpm.metadata.get_sacc_name()
        source0 = use_source_factory(
            tpm.metadata.XY.x, tpm.metadata.XY.x_measurement, tp_factory=tp_factory
        )
        source1 = use_source_factory(
            tpm.metadata.XY.y, tpm.metadata.XY.y_measurement, tp_factory=tp_factory
        )
        two_point = TwoPoint(sacc_data_type, source0, source1)
        two_point.read(sacc_data)
        two_points_old.append(two_point)

    assert len(two_points_harmonic) == len(two_points_old)

    for two_point, two_point_old in zip(two_points_harmonic, two_points_old):
        assert two_point.sacc_tracers == two_point_old.sacc_tracers
        assert two_point.sacc_data_type == two_point_old.sacc_data_type
        assert (
            two_point_old.ells is not None
        )  # The old constructor always sets the ells
        assert_array_equal(two_point.get_data_vector(), two_point_old.get_data_vector())
        assert two_point.window is not None
        assert two_point_old.window is not None
        assert_array_equal(two_point.window, two_point_old.window)
        assert two_point.ells is not None
        assert two_point_old.ells is not None
        assert_array_equal(two_point.ells, two_point_old.ells)
        assert two_point.sacc_indices is not None
        assert two_point_old.sacc_indices is not None
        assert_array_equal(two_point.sacc_indices, two_point_old.sacc_indices)

        two_point.update(params)
        two_point_old.update(params)

        assert_array_equal(
            two_point.compute_theory_vector(tools_with_vanilla_cosmology),
            two_point_old.compute_theory_vector(tools_with_vanilla_cosmology),
        )


def test_compare_constructors_reals(
    sacc_galaxy_xis, tp_factory, tools_with_vanilla_cosmology
):
    sacc_data, _, _ = sacc_galaxy_xis

    two_point_xis = extract_all_real_data(sacc_data)

    two_points_real = TwoPoint.from_measurement(two_point_xis, tp_factory)

    params = ParamsMap(two_points_real.required_parameters().get_default_values())

    two_points_old = []
    for tpm in two_point_xis:
        sacc_data_type = tpm.metadata.get_sacc_name()
        source0 = use_source_factory(
            tpm.metadata.XY.x, tpm.metadata.XY.x_measurement, tp_factory=tp_factory
        )
        source1 = use_source_factory(
            tpm.metadata.XY.y, tpm.metadata.XY.y_measurement, tp_factory=tp_factory
        )
        two_point = TwoPoint(sacc_data_type, source0, source1)
        two_point.read(sacc_data)
        two_points_old.append(two_point)

    assert len(two_points_real) == len(two_points_old)

    for two_point, two_point_old in zip(two_points_real, two_points_old):
        assert two_point.sacc_tracers == two_point_old.sacc_tracers
        assert two_point.sacc_data_type == two_point_old.sacc_data_type
        assert two_point.thetas is not None
        assert two_point_old.thetas is not None
        assert_array_equal(two_point.thetas, two_point_old.thetas)
        assert_array_equal(two_point.get_data_vector(), two_point_old.get_data_vector())
        assert two_point.sacc_indices is not None
        assert two_point_old.sacc_indices is not None
        assert_array_equal(two_point.sacc_indices, two_point_old.sacc_indices)

        two_point.update(params)
        two_point_old.update(params)

        assert_array_equal(
            two_point.compute_theory_vector(tools_with_vanilla_cosmology),
            two_point_old.compute_theory_vector(tools_with_vanilla_cosmology),
        )


def test_extract_all_data_harmonic_no_cov(sacc_galaxy_cells):
    sacc_data, _, _ = sacc_galaxy_cells
    sacc_data.covariance = None
    with pytest.raises(
        ValueError,
        match=("The SACC object does not have a dense covariance matrix."),
    ):
        _ = extract_all_harmonic_data(sacc_data, include_maybe_types=True)


def test_extract_all_data_real_no_cov(sacc_galaxy_xis):
    sacc_data, _, _ = sacc_galaxy_xis
    sacc_data.covariance = None
    with pytest.raises(
        ValueError,
        match=("The SACC object does not have a dense covariance matrix."),
    ):
        _ = extract_all_real_data(sacc_data, include_maybe_types=True)


def test_make_all_photoz_bin_combinations_with_cmb_basic():
    """Test basic functionality of make_all_photoz_bin_combinations_with_cmb."""
    # Create test galaxy bins
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
        InferredGalaxyZDist(
            bin_name="bin_2",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.SHEAR_E},
        ),
    ]

    # Test without CMB auto-correlation
    combinations = make_all_photoz_bin_combinations_with_cmb(
        galaxy_bins, include_cmb_auto=False
    )

    # Should have galaxy-galaxy combinations + CMB-galaxy cross-correlations
    galaxy_only_combinations = make_all_photoz_bin_combinations(galaxy_bins)

    # Expected: 3 galaxy combinations + 4 CMB-galaxy combinations
    # (2 each direction for compatible measurements)
    # bin_1 (COUNTS) is compatible with CMB.CONVERGENCE -> 2 combinations
    # bin_2 (SHEAR_E) is compatible with CMB.CONVERGENCE -> 2 combinations
    expected_total = len(galaxy_only_combinations) + 4

    assert len(combinations) == expected_total

    # Check that all original galaxy combinations are preserved
    galaxy_combinations_in_result = [
        combo
        for combo in combinations
        if combo.x_measurement != CMB.CONVERGENCE
        and combo.y_measurement != CMB.CONVERGENCE
    ]
    assert len(galaxy_combinations_in_result) == len(galaxy_only_combinations)


def test_make_all_photoz_bin_combinations_with_cmb_with_auto():
    """Test make_all_photoz_bin_combinations_with_cmb with CMB auto-correlation."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    # Test with CMB auto-correlation
    combinations = make_all_photoz_bin_combinations_with_cmb(
        galaxy_bins, include_cmb_auto=True
    )

    # Check for CMB auto-correlation
    cmb_auto_combinations = [
        combo
        for combo in combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            and combo.y_measurement == CMB.CONVERGENCE
            and combo.x.bin_name == "cmb_convergence"
            and combo.y.bin_name == "cmb_convergence"
        )
    ]

    assert len(cmb_auto_combinations) == 1
    cmb_auto = cmb_auto_combinations[0]
    assert cmb_auto.x.bin_name == "cmb_convergence"
    assert cmb_auto.y.bin_name == "cmb_convergence"


def test_make_all_photoz_bin_combinations_with_cmb_custom_tracer_name():
    """Test make_all_photoz_bin_combinations_with_cmb with custom CMB tracer name."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    custom_name = "my_cmb_tracer"
    combinations = make_all_photoz_bin_combinations_with_cmb(
        galaxy_bins, cmb_tracer_name=custom_name
    )

    # Check that CMB tracer has the custom name
    cmb_combinations = [
        combo
        for combo in combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            or combo.y_measurement == CMB.CONVERGENCE
        )
    ]

    assert len(cmb_combinations) > 0
    for combo in cmb_combinations:
        if combo.x_measurement == CMB.CONVERGENCE:
            assert combo.x.bin_name == custom_name
        if combo.y_measurement == CMB.CONVERGENCE:
            assert combo.y.bin_name == custom_name


def test_make_all_photoz_bin_combinations_with_cmb_measurement_compatibility():
    """Test that only compatible measurements create CMB-galaxy cross-correlations."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="counts_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},  # Compatible with CMB.CONVERGENCE
        ),
        InferredGalaxyZDist(
            bin_name="shear_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.SHEAR_E},  # Compatible with CMB.CONVERGENCE
        ),
    ]

    combinations = make_all_photoz_bin_combinations_with_cmb(galaxy_bins)

    # Check CMB-galaxy combinations
    cmb_galaxy_combinations = [
        combo
        for combo in combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            or combo.y_measurement == CMB.CONVERGENCE
        )
        and not (
            combo.x_measurement == CMB.CONVERGENCE
            and combo.y_measurement == CMB.CONVERGENCE
        )
    ]

    # Should have 4 combinations: 2 for each galaxy bin (both directions)
    assert len(cmb_galaxy_combinations) == 4

    # Verify specific combinations exist
    expected_combinations = [
        ("cmb_convergence", CMB.CONVERGENCE, "counts_bin", Galaxies.COUNTS),
        ("counts_bin", Galaxies.COUNTS, "cmb_convergence", CMB.CONVERGENCE),
        ("cmb_convergence", CMB.CONVERGENCE, "shear_bin", Galaxies.SHEAR_E),
        ("shear_bin", Galaxies.SHEAR_E, "cmb_convergence", CMB.CONVERGENCE),
    ]

    actual_combinations = [
        (combo.x.bin_name, combo.x_measurement, combo.y.bin_name, combo.y_measurement)
        for combo in cmb_galaxy_combinations
    ]

    for expected in expected_combinations:
        assert expected in actual_combinations


def test_make_all_photoz_bin_combinations_with_cmb_empty_input():
    """Test make_all_photoz_bin_combinations_with_cmb with empty galaxy bins."""
    combinations = make_all_photoz_bin_combinations_with_cmb([])

    # Should only return empty list since no galaxy bins
    assert len(combinations) == 0


def test_make_all_photoz_bin_combinations_with_cmb_cmb_bin_properties():
    """Test that the created CMB bin has correct properties."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    combinations = make_all_photoz_bin_combinations_with_cmb(galaxy_bins)

    # Find a combination with CMB
    cmb_combo = next(
        combo for combo in combinations if combo.x_measurement == CMB.CONVERGENCE
    )

    cmb_bin = cmb_combo.x
    assert cmb_bin.bin_name == "cmb_convergence"
    assert np.array_equal(cmb_bin.z, np.array([1100.0]))
    assert np.array_equal(cmb_bin.dndz, np.array([1.0]))
    assert cmb_bin.measurements == {CMB.CONVERGENCE}
    assert cmb_bin.type_source == TypeSource.DEFAULT


@pytest.mark.parametrize("include_auto", [True, False])
def test_make_all_photoz_bin_combinations_with_cmb_parametrized(include_auto: bool):
    """Parametrized test for CMB auto-correlation inclusion."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    combinations = make_all_photoz_bin_combinations_with_cmb(
        galaxy_bins, include_cmb_auto=include_auto
    )

    # Count CMB auto-correlations
    cmb_auto_count = sum(
        1
        for combo in combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            and combo.y_measurement == CMB.CONVERGENCE
        )
    )

    if include_auto:
        assert cmb_auto_count == 1
    else:
        assert cmb_auto_count == 0


def test_make_all_photoz_bin_combinations_with_cmb_multiple_measurements():
    """Test with galaxy bins that have multiple measurement types."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="multi_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS, Galaxies.SHEAR_E},  # Multiple measurements
        ),
    ]

    combinations = make_all_photoz_bin_combinations_with_cmb(galaxy_bins)

    # Should have CMB cross-correlations for both measurement types
    cmb_galaxy_combinations = [
        combo
        for combo in combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            or combo.y_measurement == CMB.CONVERGENCE
        )
        and not (
            combo.x_measurement == CMB.CONVERGENCE
            and combo.y_measurement == CMB.CONVERGENCE
        )
    ]

    # Should have 4 combinations: 2 directions × 2 measurement types
    assert len(cmb_galaxy_combinations) == 4

    # Verify both measurement types are present
    galaxy_measurements = {
        (
            combo.x_measurement
            if combo.y_measurement == CMB.CONVERGENCE
            else combo.y_measurement
        )
        for combo in cmb_galaxy_combinations
    }

    assert Galaxies.COUNTS in galaxy_measurements
    assert Galaxies.SHEAR_E in galaxy_measurements
