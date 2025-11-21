"""
Tests for the module firecrown.metadata_types and firecrown.metadata_functions.
"""

from dataclasses import replace
from itertools import product, chain
from unittest.mock import MagicMock
import re
import pytest
import numpy as np


import sacc
import sacc_name_mapping as snm
from firecrown.metadata_types import (
    ALL_MEASUREMENTS,
    Measurement,
    CMB,
    Clusters,
    Galaxies,
    LENS_REGEX,
    SOURCE_REGEX,
    TracerNames,
    TwoPointHarmonic,
    TwoPointReal,
    measurement_is_compatible as is_compatible,
    measurements_types,
)
from firecrown.metadata_types._compatibility import (
    _measurement_is_compatible_harmonic as is_compatible_harmonic,
    _measurement_is_compatible_real as is_compatible_real,
    _measurement_supports_harmonic as supports_harmonic,
    _measurement_supports_real as supports_real,
)
from firecrown.metadata_types._sacc_type_string import (
    _type_to_sacc_string_harmonic as harmonic,
    _type_to_sacc_string_real as real,
)
from firecrown.metadata_types._measurements import _compare_enums as compare_enums

from firecrown.metadata_functions import (
    TwoPointRealIndex,
    extract_all_measured_types,
    measurements_from_index,
)
from firecrown.metadata_functions._matching import match_name_type

from firecrown.data_types import TwoPointMeasurement
from firecrown.data_functions import (
    extract_all_real_data,
    extract_all_harmonic_data,
    check_two_point_consistence_harmonic,
    check_two_point_consistence_real,
)


def test_order_enums():
    assert compare_enums(CMB.CONVERGENCE, Clusters.COUNTS) < 0
    assert compare_enums(Clusters.COUNTS, CMB.CONVERGENCE) > 0

    assert compare_enums(CMB.CONVERGENCE, Galaxies.COUNTS) < 0
    assert compare_enums(Galaxies.COUNTS, CMB.CONVERGENCE) > 0

    assert compare_enums(Galaxies.SHEAR_E, Galaxies.SHEAR_T) < 0
    assert compare_enums(Galaxies.SHEAR_E, Galaxies.COUNTS) < 0
    assert compare_enums(Galaxies.SHEAR_T, Galaxies.COUNTS) < 0

    assert compare_enums(Galaxies.COUNTS, Galaxies.SHEAR_E) > 0

    for enumerand in ALL_MEASUREMENTS:
        assert compare_enums(enumerand, enumerand) == 0


def test_galaxies_is_shear():
    assert Galaxies.PART_OF_XI_MINUS == Galaxies.SHEAR_MINUS
    assert Galaxies.PART_OF_XI_PLUS == Galaxies.SHEAR_PLUS
    assert Galaxies.SHEAR_E.is_shear()
    assert Galaxies.SHEAR_T.is_shear()
    assert Galaxies.PART_OF_XI_MINUS.is_shear()
    assert Galaxies.PART_OF_XI_PLUS.is_shear()
    assert not Galaxies.COUNTS.is_shear()


def testcompare_enums_wrong_type():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unknown measurement type encountered "
            "(<enum 'Galaxies'>, <class 'int'>)."
        ),
    ):
        compare_enums(Galaxies.COUNTS, 1)


def test_enumeration_equality_galaxy():
    for e1, e2 in product(Galaxies, chain(CMB, Clusters)):
        assert e1 != e2


def test_enumeration_equality_cmb():
    for e1, e2 in product(CMB, chain(Galaxies, Clusters)):
        assert e1 != e2


def test_enumeration_equality_cluster():
    for e1, e2 in product(Clusters, chain(CMB, Galaxies)):
        assert e1 != e2


def test_exact_matches():
    for sacc_name, space, (enum_1, enum_2) in snm.mappings:
        if space == "ell":
            assert harmonic(enum_1, enum_2) == sacc_name
        elif space == "theta":
            assert real(enum_1, enum_2) == sacc_name
        else:
            raise ValueError(f"Illegal 'space' value {space} in testing data")


def test_translation_order_dependence():
    for a, b in product(ALL_MEASUREMENTS, ALL_MEASUREMENTS):
        assert isinstance(a, (Galaxies, CMB, Clusters))
        assert isinstance(b, (Galaxies, CMB, Clusters))
        if is_compatible_real(a, b):
            assert real(a, b) != real(b, a) if a != b else True
        if is_compatible_harmonic(a, b):
            assert harmonic(a, b) != harmonic(b, a) if a != b else True
        if (
            supports_harmonic(a)
            and supports_harmonic(b)
            and supports_real(a)
            and supports_real(b)
        ):
            assert harmonic(a, b) != real(a, b)


def test_unsupported_type_galaxy():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Untranslated Galaxy Measurement encountered"):
        Galaxies.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated Galaxy Measurement encountered"):
        Galaxies.polarization(unknown_type)


def test_unsupported_type_cmb():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Untranslated CMBMeasurement encountered"):
        CMB.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated CMBMeasurement encountered"):
        CMB.polarization(unknown_type)


def test_unsupported_type_cluster():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Untranslated ClusterMeasurement encountered"):
        Clusters.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated ClusterMeasurement encountered"):
        Clusters.polarization(unknown_type)


def test_type_hashs():
    for e1, e2 in product(ALL_MEASUREMENTS, ALL_MEASUREMENTS):
        if e1 == e2:
            assert hash(e1) == hash(e2)
        else:
            assert hash(e1) != hash(e2)


def test_measurement_is_compatible():
    for a, b in product(ALL_MEASUREMENTS, ALL_MEASUREMENTS):
        assert isinstance(a, (Galaxies, CMB, Clusters))
        assert isinstance(b, (Galaxies, CMB, Clusters))
        if is_compatible_real(a, b) or is_compatible_harmonic(a, b):
            assert is_compatible(a, b)
        else:
            assert not is_compatible(a, b)


def test_extract_all_tracers_types_cells(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cells

    tracers = extract_all_measured_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.COUNTS
        if SOURCE_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.SHEAR_E


def test_extract_all_tracers_types_cwindows(
    sacc_galaxy_cwindows: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cwindows

    tracers = extract_all_measured_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.COUNTS
        if SOURCE_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.SHEAR_E


def test_extract_all_tracers_types_reals(sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict]):
    sacc_data, _, _ = sacc_galaxy_xis

    tracers = extract_all_measured_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.COUNTS
        if SOURCE_REGEX.match(tracer):
            assert measurements == {
                Galaxies.SHEAR_T,
                Galaxies.PART_OF_XI_MINUS,
                Galaxies.PART_OF_XI_PLUS,
            }


def test_extract_all_tracers_types_reals_inverted(
    sacc_galaxy_xis_inverted: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_xis_inverted

    with pytest.warns(DeprecationWarning, match="AUTO-CORRECTION PERFORMED"):
        tracers = extract_all_measured_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            assert measurements == {Galaxies.COUNTS}
        if SOURCE_REGEX.match(tracer):
            assert measurements == {
                Galaxies.SHEAR_T,
                Galaxies.PART_OF_XI_MINUS,
                Galaxies.PART_OF_XI_PLUS,
            }


def test_extract_all_tracers_types_cells_include_maybe(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cells

    assert extract_all_measured_types(sacc_data) == extract_all_measured_types(
        sacc_data
    )


def test_extract_all_tracers_types_cwindows_include_maybe(
    sacc_galaxy_cwindows: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cwindows

    assert extract_all_measured_types(sacc_data) == extract_all_measured_types(
        sacc_data
    )


def test_extract_all_tracers_types_xi_thetas_include_maybe(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_xis

    assert extract_all_measured_types(sacc_data) == extract_all_measured_types(
        sacc_data
    )


def test_measurements_from_index1():
    index: TwoPointRealIndex = {
        "data_type": "galaxy_shearDensity_xi_t",
        "tracer_names": TracerNames("src0", "lens0"),
    }
    n1, a, n2, b = measurements_from_index(index)
    assert n1 == "src0"
    assert a == Galaxies.SHEAR_T
    assert n2 == "lens0"
    assert b == Galaxies.COUNTS


def test_measurements_from_index2():
    index: TwoPointRealIndex = {
        "data_type": "galaxy_shearDensity_xi_t",
        "tracer_names": TracerNames("lens0", "src0"),
    }
    n1, a, n2, b = measurements_from_index(index)
    assert n1 == "lens0"
    assert a == Galaxies.SHEAR_T
    assert n2 == "src0"
    assert b == Galaxies.COUNTS


def test_check_two_point_consistence_harmonic(two_point_cell: TwoPointHarmonic):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=two_point_cell,
    )
    check_two_point_consistence_harmonic([tpm])


def test_check_two_point_consistence_harmonic_real(
    optimized_two_point_real: TwoPointReal,
):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=optimized_two_point_real,
    )
    with pytest.raises(
        ValueError,
        match=(".*is not a measurement of TwoPointHarmonic."),
    ):
        check_two_point_consistence_harmonic([tpm])


def test_check_two_point_consistence_real(optimized_two_point_real: TwoPointReal):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=optimized_two_point_real,
    )
    check_two_point_consistence_real([tpm])


def test_check_two_point_consistence_real_harmonic(
    optimized_two_point_cell: TwoPointReal,
):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=optimized_two_point_cell,
    )
    with pytest.raises(
        ValueError,
        match=(".*is not a measurement of TwoPointReal."),
    ):
        check_two_point_consistence_real([tpm])


def test_check_two_point_consistence_harmonic_mixing_cov(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cells

    tpms = extract_all_harmonic_data(sacc_data)

    tpms[0] = replace(tpms[0], covariance_name="wrong_cov_name")

    with pytest.raises(
        ValueError,
        match=(
            ".* has a different covariance name .* "
            "than the previous TwoPointHarmonic wrong_cov_name."
        ),
    ):
        check_two_point_consistence_harmonic(tpms)


def test_check_two_point_consistence_real_mixing_cov(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_xis

    tpms = extract_all_real_data(sacc_data)
    tpms[0] = replace(tpms[0], covariance_name="wrong_cov_name")

    with pytest.raises(
        ValueError,
        match=(
            ".* has a different covariance name .* than the "
            "previous TwoPointReal wrong_cov_name."
        ),
    ):
        check_two_point_consistence_real(tpms)


def test_check_two_point_consistence_harmonic_non_unique_indices(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cells

    tpms = extract_all_harmonic_data(sacc_data)

    new_indices = tpms[0].indices
    new_indices[0] = 3
    tpms[0] = replace(tpms[0], indices=new_indices)

    with pytest.raises(
        ValueError,
        match=".* are not unique.",
    ):
        check_two_point_consistence_harmonic(tpms)


def test_check_two_point_consistence_real_non_unique_indices(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_xis

    tpms = extract_all_real_data(sacc_data)
    new_indices = tpms[0].indices
    new_indices[0] = 3
    tpms[0] = replace(tpms[0], indices=new_indices)

    with pytest.raises(
        ValueError,
        match=".* are not unique.",
    ):
        check_two_point_consistence_real(tpms)


def test_check_two_point_consistence_harmonic_indices_overlap(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cells

    tpms = extract_all_harmonic_data(sacc_data)

    new_indices = tpms[1].indices
    new_indices[1] = 3
    tpms[1] = replace(tpms[1], indices=new_indices)

    with pytest.raises(
        ValueError,
        match=".* overlap.",
    ):
        check_two_point_consistence_harmonic(tpms)


def test_check_two_point_consistence_real_indices_overlap(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_xis

    tpms = extract_all_real_data(sacc_data)

    new_indices = tpms[1].indices
    new_indices[1] = 3
    tpms[1] = replace(tpms[1], indices=new_indices)

    with pytest.raises(
        ValueError,
        match=".* overlap.",
    ):
        check_two_point_consistence_real(tpms)


def test_extract_all_data_cells_order_convention(
    sacc_galaxy_cells_order_convention: sacc.Sacc,
):
    two_point_list = extract_all_harmonic_data(sacc_galaxy_cells_order_convention)
    assert len(two_point_list) == 1

    assert two_point_list[0].metadata.XY.x.bin_name == "bin0"
    assert two_point_list[0].metadata.XY.y.bin_name == "bin1"


def test_measurements_types_single_galaxy_lens():
    """Test measurements_types with single Galaxy lens measurement type."""
    measurements: set[Measurement] = {Galaxies.COUNTS}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is False
    assert types == ["Galaxy lens"]


def test_measurements_types_single_galaxy_source():
    """Test measurements_types with single Galaxy source measurement type."""
    measurements: set[Measurement] = {Galaxies.SHEAR_E}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is False
    assert types == ["Galaxy source"]


def test_measurements_types_single_cmb():
    """Test measurements_types with single CMB measurement type."""
    measurements: set[Measurement] = {CMB.CONVERGENCE}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is False
    assert types == ["CMB"]


def test_measurements_types_single_cluster():
    """Test measurements_types with single Cluster measurement type."""
    measurements: set[Measurement] = {Clusters.COUNTS}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is False
    assert types == ["Clusters"]


def test_measurements_types_multiple_galaxy_source():
    """Test measurements_types with multiple Galaxy source measurements (same type)."""
    measurements: set[Measurement] = {
        Galaxies.SHEAR_E,
        Galaxies.SHEAR_T,
        Galaxies.PART_OF_XI_PLUS,
    }
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is False
    assert types == ["Galaxy source"]


def test_measurements_types_galaxy_lens_and_source():
    """Test measurements_types with both Galaxy lens and source (mixed types)."""
    measurements: set[Measurement] = {Galaxies.COUNTS, Galaxies.SHEAR_E}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is True
    assert set(types) == {"Galaxy lens", "Galaxy source"}


def test_measurements_types_galaxy_and_cmb():
    """Test measurements_types with Galaxy and CMB (mixed types)."""
    measurements: set[Measurement] = {Galaxies.COUNTS, CMB.CONVERGENCE}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is True
    assert set(types) == {"Galaxy lens", "CMB"}


def test_measurements_types_galaxy_and_cluster():
    """Test measurements_types with Galaxy and Cluster (mixed types)."""
    measurements: set[Measurement] = {Galaxies.SHEAR_E, Clusters.COUNTS}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is True
    assert set(types) == {"Galaxy source", "Clusters"}


def test_measurements_types_cmb_and_cluster():
    """Test measurements_types with CMB and Cluster (mixed types)."""
    measurements: set[Measurement] = {CMB.CONVERGENCE, Clusters.COUNTS}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is True
    assert set(types) == {"CMB", "Clusters"}


def test_measurements_types_all_three_types():
    """Test measurements_types with Galaxy, CMB, and Cluster (three types)."""
    measurements: set[Measurement] = {Galaxies.COUNTS, CMB.CONVERGENCE, Clusters.COUNTS}
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is True
    assert set(types) == {"Galaxy lens", "CMB", "Clusters"}


def test_measurements_types_all_four_categories():
    """Test measurements_types with Galaxy lens, Galaxy source, CMB, and Cluster."""
    measurements: set[Measurement] = {
        Galaxies.COUNTS,  # Galaxy lens
        Galaxies.SHEAR_E,  # Galaxy source
        CMB.CONVERGENCE,  # CMB
        Clusters.COUNTS,  # Clusters
    }
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is True
    assert set(types) == {"Galaxy lens", "Galaxy source", "CMB", "Clusters"}


def test_measurements_types_empty_set():
    """Test measurements_types with empty measurement set."""
    measurements: set[Measurement] = set()
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is False
    assert not types


def test_measurements_types_order_independent():
    """Test that measurements_types returns consistent results.

    This test checks that measurements_types returns the same results regardless of set
    order.
    """
    measurements1: set[Measurement] = {Galaxies.COUNTS, CMB.CONVERGENCE}
    measurements2: set[Measurement] = {CMB.CONVERGENCE, Galaxies.COUNTS}

    has_mixed1, types1 = measurements_types(measurements1)
    has_mixed2, types2 = measurements_types(measurements2)

    assert has_mixed1 == has_mixed2
    assert set(types1) == set(types2)


def test_measurements_types_multiple_galaxy_and_cmb():
    """Test measurements_types with multiple Galaxy measurements and CMB."""
    measurements: set[Measurement] = {
        Galaxies.COUNTS,  # Galaxy lens
        Galaxies.SHEAR_E,  # Galaxy source
        Galaxies.SHEAR_T,  # Galaxy source
        CMB.CONVERGENCE,  # CMB
    }
    has_mixed, types = measurements_types(measurements)

    assert has_mixed is True
    assert set(types) == {"Galaxy lens", "Galaxy source", "CMB"}
