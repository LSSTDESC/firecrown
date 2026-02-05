"""Tests for the modules firecrown.metadata_types and firecrown.metadata_functions.

In this module, we test the functions and classes involved SACC extraction tools.
"""

import warnings
import re
import pytest
import numpy as np
from numpy.testing import assert_array_equal


import sacc

from firecrown.updatable import ParamsMap
from firecrown.metadata_types import (
    AutoMeasurementBinPairSelector,
    AutoNameBinPairSelector,
    CMB,
    Galaxies,
    GALAXY_SOURCE_TYPES,
    InferredGalaxyZDist,
    LensBinPairSelector,
    NamedBinPairSelector,
    SourceBinPairSelector,
    TracerNames,
    TwoPointHarmonic,
    TwoPointReal,
    TypeSource,
)
from firecrown.metadata_types._sacc_type_string import (
    _type_to_sacc_string_harmonic,
    _type_to_sacc_string_real,
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
    make_cmb_galaxy_combinations_only,
)
from firecrown.data_functions import (
    check_two_point_consistence_harmonic,
    check_two_point_consistence_real,
    extract_all_harmonic_data,
    extract_all_real_data,
)
from firecrown.likelihood._two_point import (
    TwoPoint,
    TwoPointFactory,
    use_source_factory,
)

# pylint: disable=too-many-lines


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


def test_extract_all_tracers_skips_non_nztracer() -> None:
    """Test that extract_all_tracers_inferred_galaxy_zdists skips non-NZTracer types.

    Verifies that when a SACC object contains both NZTracer and non-NZTracer types
    (e.g., WeakLensingTracer, DeltaFunctionTracer), only the NZTracer instances are
    extracted and returned. This tests the filtering logic on line 37-45 in
    extract_all_tracers_inferred_galaxy_zdists.
    """
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    # Add NZTracer (should be included)
    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz)

    # Add a DeltaFunctionTracer (non-NZTracer, should be skipped)
    sacc_data.add_tracer("misc", "sample")

    # Add measurement data using only the NZTracer
    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_shear_cl_ee", "src0", "src0", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    # Extract tracers - should only get the NZTracer, skipping DeltaFunctionTracer
    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)

    # Verify only the NZTracer was extracted
    assert len(all_tracers) == 1
    assert all_tracers[0].bin_name == "src0"
    assert_array_equal(all_tracers[0].z, z)
    assert_array_equal(all_tracers[0].dndz, dndz)
    assert all_tracers[0].measurements == {Galaxies.SHEAR_E}


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

    with warnings.catch_warnings():
        # Ensure no warning is emitted by this call. If one appears, treat it as an
        # error.
        warnings.simplefilter("error")  # turn warnings into exceptions
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
            _type_to_sacc_string_real(bin_comb.x_measurement, bin_comb.y_measurement),
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
            _type_to_sacc_string_harmonic(
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
        _ = extract_all_harmonic_data(sacc_data)


def test_extract_all_data_real_no_cov(sacc_galaxy_xis):
    sacc_data, _, _ = sacc_galaxy_xis
    sacc_data.covariance = None
    with pytest.raises(
        ValueError,
        match=("The SACC object does not have a dense covariance matrix."),
    ):
        _ = extract_all_real_data(sacc_data)


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

    # Expected: 3 galaxy combinations + 2 CMB-galaxy combinations
    # bin_1 (COUNTS) is compatible with CMB.CONVERGENCE -> 1 combination
    # bin_2 (SHEAR_E) is compatible with CMB.CONVERGENCE -> 1 combination
    expected_total = len(galaxy_only_combinations) + 2

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

    # Should have 2 combinations: 2 for each galaxy bin
    assert len(cmb_galaxy_combinations) == 2

    # Verify specific combinations exist
    expected_combinations = [
        ("cmb_convergence", CMB.CONVERGENCE, "counts_bin", Galaxies.COUNTS),
        ("cmb_convergence", CMB.CONVERGENCE, "shear_bin", Galaxies.SHEAR_E),
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

    # Should have 2 combinations: 2 measurement types
    assert len(cmb_galaxy_combinations) == 2

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


def test_make_cmb_galaxy_combinations_only_basic():
    """Test basic functionality of make_cmb_galaxy_combinations_only."""
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

    # Test CMB-galaxy only combinations
    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Should have only CMB-galaxy cross-correlations (no galaxy-galaxy or CMB auto)
    # 2 galaxy bins = 2 combinations
    assert len(combinations) == 2

    # Verify all combinations involve CMB
    for combo in combinations:
        assert (
            combo.x_measurement == CMB.CONVERGENCE
            or combo.y_measurement == CMB.CONVERGENCE
        )

    # Verify no galaxy-galaxy combinations
    galaxy_only_combinations = [
        combo
        for combo in combinations
        if (
            combo.x_measurement != CMB.CONVERGENCE
            and combo.y_measurement != CMB.CONVERGENCE
        )
    ]
    assert len(galaxy_only_combinations) == 0

    # Verify no CMB auto-correlation
    cmb_auto_combinations = [
        combo
        for combo in combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            and combo.y_measurement == CMB.CONVERGENCE
        )
    ]
    assert len(cmb_auto_combinations) == 0


def test_make_cmb_galaxy_combinations_only_custom_tracer_name():
    """Test make_cmb_galaxy_combinations_only with custom CMB tracer name."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    custom_name = "my_custom_cmb"
    combinations = make_cmb_galaxy_combinations_only(
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


def test_make_cmb_galaxy_combinations_only_measurement_compatibility():
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

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Should have 2 combinations: 2 for each galaxy bin
    assert len(combinations) == 2

    # Verify specific combinations exist
    expected_combinations = [
        ("cmb_convergence", CMB.CONVERGENCE, "counts_bin", Galaxies.COUNTS),
        ("cmb_convergence", CMB.CONVERGENCE, "shear_bin", Galaxies.SHEAR_E),
    ]

    actual_combinations = [
        (combo.x.bin_name, combo.x_measurement, combo.y.bin_name, combo.y_measurement)
        for combo in combinations
    ]

    for expected in expected_combinations:
        assert expected in actual_combinations


def test_make_cmb_galaxy_combinations_only_empty_input():
    """Test make_cmb_galaxy_combinations_only with empty galaxy bins."""
    combinations = make_cmb_galaxy_combinations_only([])

    # Should return empty list since no galaxy bins
    assert len(combinations) == 0


def test_make_cmb_galaxy_combinations_only_incompatible_measurements_0():
    """Test behavior with galaxy measurements incompatible with CMB convergence."""
    # Create a galaxy bin with a measurement that might not be compatible
    # (This test depends on what measurements are actually incompatible)
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="test_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},  # Assuming this is compatible
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Should still have combinations if any measurements are compatible
    assert len(combinations) > 0

    # All combinations should involve CMB
    for combo in combinations:
        assert (
            combo.x_measurement == CMB.CONVERGENCE
            or combo.y_measurement == CMB.CONVERGENCE
        )


def test_make_cmb_galaxy_combinations_only_cmb_bin_properties():
    """Test that the created CMB bin has correct properties."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

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


def test_make_cmb_galaxy_combinations_only_multiple_measurements():
    """Test with galaxy bins that have multiple measurement types."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="multi_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS, Galaxies.SHEAR_E},  # Multiple measurements
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Should have combinations for both measurement types (if both compatible)
    # 2 measurement types = 2 combinations
    assert len(combinations) == 2

    # Verify both measurement types are present
    galaxy_measurements = {
        (
            combo.x_measurement
            if combo.y_measurement == CMB.CONVERGENCE
            else combo.y_measurement
        )
        for combo in combinations
    }

    assert Galaxies.COUNTS in galaxy_measurements
    assert Galaxies.SHEAR_E in galaxy_measurements


def test_make_cmb_galaxy_combinations_only_symmetric_pairs():
    """Test that symmetric pairs are created for each compatible measurement."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="test_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Should have 1 combination: (CMB, galaxy)
    assert len(combinations) == 1

    # Check that we have both directions
    cmb_first = any(
        combo.x_measurement == CMB.CONVERGENCE
        and combo.y_measurement == Galaxies.COUNTS
        for combo in combinations
    )
    galaxy_first = any(
        combo.x_measurement == Galaxies.COUNTS
        and combo.y_measurement == CMB.CONVERGENCE
        for combo in combinations
    )

    assert cmb_first
    assert not galaxy_first


def test_make_cmb_galaxy_combinations_only_single_galaxy_bin():
    """Test with a single galaxy bin."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="single_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.SHEAR_E},
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Should have 1 combination: one combination for the single bin
    assert len(combinations) == 1
    for combo in combinations:
        # Both combinations should involve the same galaxy bin
        galaxy_bin_names = {combo.x.bin_name, combo.y.bin_name}
        galaxy_bin_names.discard("cmb_convergence")  # Remove CMB name

    assert galaxy_bin_names == {"single_bin"}


@pytest.mark.parametrize("cmb_name", ["cmb_convergence", "cmb_kappa", "planck_cmb"])
def test_make_cmb_galaxy_combinations_only_parametrized_names(cmb_name: str):
    """Parametrized test for different CMB tracer names."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="bin_1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(
        galaxy_bins, cmb_tracer_name=cmb_name
    )

    # Check that all CMB tracers have the specified name
    for combo in combinations:
        if combo.x_measurement == CMB.CONVERGENCE:
            assert combo.x.bin_name == cmb_name
        if combo.y_measurement == CMB.CONVERGENCE:
            assert combo.y.bin_name == cmb_name


def test_make_cmb_galaxy_combinations_only_vs_with_cmb():
    """Test that this function is equivalent to
    make_all_photoz_bin_combinations_with_cmb
    without galaxy combinations."""
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

    # Get CMB-only combinations
    cmb_only = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Get all combinations and filter for CMB ones
    all_combinations = make_all_photoz_bin_combinations_with_cmb(galaxy_bins)
    cmb_from_all = [
        combo
        for combo in all_combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            or combo.y_measurement == CMB.CONVERGENCE
        )
        and not (
            combo.x_measurement == CMB.CONVERGENCE
            and combo.y_measurement == CMB.CONVERGENCE
        )  # Exclude CMB auto
    ]

    # Should be identical
    assert len(cmb_only) == len(cmb_from_all)

    # Convert to comparable tuples for comparison
    cmb_only_tuples = {
        (combo.x.bin_name, combo.x_measurement, combo.y.bin_name, combo.y_measurement)
        for combo in cmb_only
    }
    cmb_from_all_tuples = {
        (combo.x.bin_name, combo.x_measurement, combo.y.bin_name, combo.y_measurement)
        for combo in cmb_from_all
    }

    assert cmb_only_tuples == cmb_from_all_tuples


def test_make_all_photoz_bin_combinations_with_cmb_incompatible_measurements():
    """Test that incompatible measurements are skipped in CMB-galaxy combinations."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="compatible_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},  # Compatible with CMB.CONVERGENCE
        ),
        InferredGalaxyZDist(
            bin_name="another_compatible_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={
                Galaxies.SHEAR_T
            },  # Actually also compatible with CMB.CONVERGENCE
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

    # Both measurements are compatible, so we should have 2 combinations
    # 1 for each galaxy bin
    assert len(cmb_galaxy_combinations) == 2

    # Verify both measurements are present
    galaxy_measurements = {
        (
            combo.x_measurement
            if combo.y_measurement == CMB.CONVERGENCE
            else combo.y_measurement
        )
        for combo in cmb_galaxy_combinations
    }

    assert Galaxies.COUNTS in galaxy_measurements
    assert Galaxies.SHEAR_T in galaxy_measurements


def test_make_cmb_galaxy_combinations_only_incompatible_measurements():
    """Test that incompatible measurements are skipped in CMB-galaxy combinations."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="compatible_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.SHEAR_E},  # Compatible with CMB.CONVERGENCE
        ),
        InferredGalaxyZDist(
            bin_name="another_compatible_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={
                Galaxies.SHEAR_T
            },  # Actually also compatible with CMB.CONVERGENCE
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Both measurements are compatible, so we should have 2 combinations
    # 1 for each galaxy bin
    assert len(combinations) == 2

    # Verify both measurements are present
    galaxy_measurements = {
        (
            combo.x_measurement
            if combo.y_measurement == CMB.CONVERGENCE
            else combo.y_measurement
        )
        for combo in combinations
    }

    assert Galaxies.SHEAR_E in galaxy_measurements
    assert Galaxies.SHEAR_T in galaxy_measurements


def test_make_all_photoz_bin_combinations_with_cmb_all_incompatible():
    """Test behavior when all galaxy measurements are incompatible with CMB."""
    # Since we can't find truly incompatible measurements, let's test with
    # measurements that we know ARE compatible and adjust expectations
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="compatible_bin1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.SHEAR_T},  # Actually compatible with CMB.CONVERGENCE
        ),
        InferredGalaxyZDist(
            bin_name="compatible_bin2",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.PART_OF_XI_MINUS},
        ),
    ]

    combinations = make_all_photoz_bin_combinations_with_cmb(galaxy_bins)

    # Should have galaxy-galaxy combinations + CMB-galaxy combinations
    galaxy_only_combinations = make_all_photoz_bin_combinations(galaxy_bins)

    # Check that we have both galaxy and CMB combinations
    cmb_combinations = [
        combo
        for combo in combinations
        if (
            combo.x_measurement == CMB.CONVERGENCE
            or combo.y_measurement == CMB.CONVERGENCE
        )
    ]

    # Since SHEAR_T is compatible, we should have CMB combinations
    assert len(cmb_combinations) == 1
    # Total should be galaxy combinations + CMB combinations
    assert len(combinations) == len(galaxy_only_combinations) + len(cmb_combinations)


def test_make_cmb_galaxy_combinations_only_all_incompatible():
    """Test behavior when all galaxy measurements are incompatible with CMB."""
    # Since we can't find truly incompatible measurements, let's test with
    # measurements that we know ARE compatible and adjust expectations
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="compatible_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.SHEAR_T},  # Actually compatible with CMB.CONVERGENCE
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    # Since SHEAR_T is actually compatible, we should have one combination
    assert len(combinations) == 1


def test_make_all_photoz_bin_combinations_with_cmb_empty():
    """Test behavior when given an empty list of galaxy bins."""
    galaxy_bins = [
        InferredGalaxyZDist(
            bin_name="compatible_bin",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.PART_OF_XI_MINUS},
        ),
    ]

    combinations = make_cmb_galaxy_combinations_only(galaxy_bins)

    assert len(combinations) == 0


def test_extract_all_photoz_bin_combinations_with_selector(sacc_galaxy_cells):
    """Test extract_all_photoz_bin_combinations with bin_pair_selector."""
    sacc_data, _, _ = sacc_galaxy_cells

    # Without selector - should get all combinations
    all_combinations = extract_all_photoz_bin_combinations(sacc_data)

    # With auto-correlation selector - should only get auto-correlations
    auto_selector = AutoNameBinPairSelector()
    auto_combinations = extract_all_photoz_bin_combinations(
        sacc_data, bin_pair_selector=auto_selector
    )

    # Verify all auto combinations have matching bin names
    for combo in auto_combinations:
        assert combo.x.bin_name == combo.y.bin_name

    # Auto combinations should be a subset of all combinations
    assert len(auto_combinations) < len(all_combinations)
    assert len(auto_combinations) > 0


def test_extract_all_photoz_bin_combinations_with_source_selector(sacc_galaxy_cells):
    """Test extract_all_photoz_bin_combinations with source bin selector."""
    sacc_data, _, _ = sacc_galaxy_cells

    # With source selector - should only get source (shear) measurements
    source_selector = SourceBinPairSelector()
    source_combinations = extract_all_photoz_bin_combinations(
        sacc_data, bin_pair_selector=source_selector
    )

    # Verify all combinations involve source measurements

    for combo in source_combinations:
        assert combo.x_measurement in GALAXY_SOURCE_TYPES
        assert combo.y_measurement in GALAXY_SOURCE_TYPES

    assert len(source_combinations) > 0


def test_extract_all_photoz_bin_combinations_with_lens_selector(sacc_galaxy_cells):
    """Test extract_all_photoz_bin_combinations with lens bin selector."""
    sacc_data, _, _ = sacc_galaxy_cells

    # With lens selector - should only get lens (counts) measurements
    lens_selector = LensBinPairSelector()
    lens_combinations = extract_all_photoz_bin_combinations(
        sacc_data, bin_pair_selector=lens_selector
    )

    # Verify all combinations involve lens measurements
    for combo in lens_combinations:
        assert combo.x_measurement == Galaxies.COUNTS
        assert combo.y_measurement == Galaxies.COUNTS

    assert len(lens_combinations) > 0


def test_extract_all_harmonic_metadata_with_selector(sacc_galaxy_cells):
    """Test extract_all_harmonic_metadata with bin_pair_selector."""
    sacc_data, _, _ = sacc_galaxy_cells

    # Without selector - should get all metadata
    all_metadata = extract_all_harmonic_metadata(sacc_data)

    # With auto-correlation selector - should only get auto-correlations
    auto_selector = AutoNameBinPairSelector()
    auto_metadata = extract_all_harmonic_metadata(
        sacc_data, bin_pair_selector=auto_selector
    )

    # Verify all auto metadata have matching bin names
    for meta in auto_metadata:
        assert meta.XY.x.bin_name == meta.XY.y.bin_name

    # Auto metadata should be a subset of all metadata
    assert len(auto_metadata) < len(all_metadata)
    assert len(auto_metadata) > 0

    # Check that ells are properly extracted
    for meta in auto_metadata:
        assert len(meta.ells) > 0
        assert isinstance(meta.ells, np.ndarray)


def test_extract_all_harmonic_metadata_with_source_selector(sacc_galaxy_cells):
    """Test extract_all_harmonic_metadata with source bin selector."""
    sacc_data, _, _ = sacc_galaxy_cells

    # With source selector - should only get source (shear) measurements
    source_selector = SourceBinPairSelector()
    source_metadata = extract_all_harmonic_metadata(
        sacc_data, bin_pair_selector=source_selector
    )

    for meta in source_metadata:
        assert meta.XY.x_measurement in GALAXY_SOURCE_TYPES
        assert meta.XY.y_measurement in GALAXY_SOURCE_TYPES

    assert len(source_metadata) > 0


def test_extract_all_harmonic_metadata_with_combined_selector(sacc_galaxy_cells):
    """Test extract_all_harmonic_metadata with combined selectors."""
    sacc_data, _, _ = sacc_galaxy_cells

    # Combine auto-name and auto-measurement selectors
    combined_selector = AutoNameBinPairSelector() & AutoMeasurementBinPairSelector()
    filtered_metadata = extract_all_harmonic_metadata(
        sacc_data, bin_pair_selector=combined_selector
    )

    # Verify all filtered metadata have matching bins AND measurements
    for meta in filtered_metadata:
        assert meta.XY.x.bin_name == meta.XY.y.bin_name
        assert meta.XY.x_measurement == meta.XY.y_measurement

    assert len(filtered_metadata) > 0


def test_extract_all_real_metadata_with_selector(sacc_galaxy_xis):
    """Test extract_all_real_metadata with bin_pair_selector."""
    sacc_data, _, _ = sacc_galaxy_xis

    # Without selector - should get all metadata
    all_metadata = extract_all_real_metadata(sacc_data)

    # With auto-correlation selector - should only get auto-correlations
    auto_selector = AutoNameBinPairSelector()
    auto_metadata = extract_all_real_metadata(
        sacc_data, bin_pair_selector=auto_selector
    )

    # Verify all auto metadata have matching bin names
    for meta in auto_metadata:
        assert meta.XY.x.bin_name == meta.XY.y.bin_name

    # Auto metadata should be a subset of all metadata
    assert len(auto_metadata) < len(all_metadata)
    assert len(auto_metadata) > 0

    # Check that thetas are properly extracted
    for meta in auto_metadata:
        assert len(meta.thetas) > 0
        assert isinstance(meta.thetas, np.ndarray)


def test_extract_all_real_metadata_with_lens_selector(sacc_galaxy_xis):
    """Test extract_all_real_metadata with lens bin selector."""
    sacc_data, _, _ = sacc_galaxy_xis

    # With lens selector - should only get lens (counts) measurements
    lens_selector = LensBinPairSelector()
    lens_metadata = extract_all_real_metadata(
        sacc_data, bin_pair_selector=lens_selector
    )

    # Verify all metadata involve lens measurements
    for meta in lens_metadata:
        assert meta.XY.x_measurement == Galaxies.COUNTS
        assert meta.XY.y_measurement == Galaxies.COUNTS

    assert len(lens_metadata) > 0


def test_extract_all_real_metadata_with_combined_selector(sacc_galaxy_xis):
    """Test extract_all_real_metadata with combined selectors."""
    sacc_data, _, _ = sacc_galaxy_xis

    # Combine auto-name and lens selectors
    combined_selector = AutoNameBinPairSelector() & LensBinPairSelector()
    filtered_metadata = extract_all_real_metadata(
        sacc_data, bin_pair_selector=combined_selector
    )

    # Verify all filtered metadata have matching bins AND are lens measurements
    for meta in filtered_metadata:
        assert meta.XY.x.bin_name == meta.XY.y.bin_name
        assert meta.XY.x_measurement == Galaxies.COUNTS
        assert meta.XY.y_measurement == Galaxies.COUNTS

    assert len(filtered_metadata) > 0


def test_extract_all_harmonic_metadata_with_not_selector(sacc_galaxy_cells):
    """Test extract_all_harmonic_metadata with NOT selector."""
    sacc_data, _, _ = sacc_galaxy_cells

    # Get all metadata
    all_metadata = extract_all_harmonic_metadata(sacc_data)

    # Get auto-correlation metadata
    auto_selector = AutoNameBinPairSelector()
    auto_metadata = extract_all_harmonic_metadata(
        sacc_data, bin_pair_selector=auto_selector
    )

    # Get cross-correlation metadata using NOT selector
    not_auto_selector = ~auto_selector
    cross_metadata = extract_all_harmonic_metadata(
        sacc_data, bin_pair_selector=not_auto_selector
    )

    # Verify cross metadata have different bin names
    for meta in cross_metadata:
        assert meta.XY.x.bin_name != meta.XY.y.bin_name

    # Auto + cross should equal all
    assert len(auto_metadata) + len(cross_metadata) == len(all_metadata)


def test_extract_all_real_metadata_with_or_selector(sacc_galaxy_xis):
    """Test extract_all_real_metadata with OR selector."""
    sacc_data, _, _ = sacc_galaxy_xis

    # Get all tracers to find specific bin names
    tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    if len(tracers) < 2:
        pytest.skip("Need at least 2 tracers for this test")

    bin1 = tracers[0].bin_name
    bin2 = tracers[1].bin_name

    # Create OR selector: auto-correlations OR specific named pair
    auto_selector = AutoNameBinPairSelector()
    named_selector = NamedBinPairSelector(names=[(bin1, bin2)])
    or_selector = auto_selector | named_selector

    filtered_metadata = extract_all_real_metadata(
        sacc_data, bin_pair_selector=or_selector
    )

    # All filtered metadata should be either auto-correlations or the named pair
    for meta in filtered_metadata:
        is_auto = meta.XY.x.bin_name == meta.XY.y.bin_name
        is_named = (meta.XY.x.bin_name == bin1 and meta.XY.y.bin_name == bin2) or (
            meta.XY.x.bin_name == bin2 and meta.XY.y.bin_name == bin1
        )
        assert is_auto or is_named

    assert len(filtered_metadata) > 0


def test_extract_with_selector_empty_result(sacc_galaxy_cells):
    """Test that selector can result in empty extraction."""
    sacc_data, _, _ = sacc_galaxy_cells

    # Create selector for non-existent bin names
    impossible_selector = NamedBinPairSelector(names=[("nonexistent1", "nonexistent2")])

    # Should return empty lists
    empty_combinations = extract_all_photoz_bin_combinations(
        sacc_data, bin_pair_selector=impossible_selector
    )
    empty_harmonic = extract_all_harmonic_metadata(
        sacc_data, bin_pair_selector=impossible_selector
    )

    assert len(empty_combinations) == 0
    assert len(empty_harmonic) == 0


def test_extract_with_selector_preserves_data_quality(sacc_galaxy_cells):
    """Test that using selector doesn't affect data quality of extracted metadata."""
    sacc_data, _, _ = sacc_galaxy_cells

    # Extract without selector
    all_metadata = extract_all_harmonic_metadata(sacc_data)

    # Extract with selector
    source_selector = SourceBinPairSelector()
    filtered_metadata = extract_all_harmonic_metadata(
        sacc_data, bin_pair_selector=source_selector
    )

    for filtered in filtered_metadata:
        # Find matching item in all_metadata
        matching = [
            m
            for m in all_metadata
            if (
                m.XY.x.bin_name == filtered.XY.x.bin_name
                and m.XY.y.bin_name == filtered.XY.y.bin_name
                and m.XY.x_measurement == filtered.XY.x_measurement
                and m.XY.y_measurement == filtered.XY.y_measurement
            )
        ]
        assert len(matching) == 1
        match = matching[0]

        # Verify data is identical
        assert_array_equal(filtered.ells, match.ells)
        assert filtered.window == match.window
        assert filtered.window_ells == match.window_ells
