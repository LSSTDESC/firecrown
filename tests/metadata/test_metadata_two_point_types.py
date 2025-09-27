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
    CMB,
    Clusters,
    Galaxies,
    LENS_REGEX,
    SOURCE_REGEX,
    TracerNames,
    TwoPointHarmonic,
    TwoPointReal,
    compare_enums,
    measurement_is_compatible as is_compatible,
    measurement_is_compatible_harmonic as is_compatible_harmonic,
    measurement_is_compatible_real as is_compatible_real,
    measurement_supports_harmonic as supports_harmonic,
    measurement_supports_real as supports_real,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
)

from firecrown.metadata_functions import (
    TwoPointRealIndex,
    extract_all_measured_types,
    match_name_type,
    measurements_from_index,
)

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


def test_compare_enums_wrong_type():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unknown measurement type encountered "
            "(<enum 'Galaxies'>, <class 'int'>)."
        ),
    ):
        compare_enums(Galaxies.COUNTS, 1)  # type: ignore


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


def test_translation_invariants():
    for a, b in product(ALL_MEASUREMENTS, ALL_MEASUREMENTS):
        assert isinstance(a, (Galaxies, CMB, Clusters))
        assert isinstance(b, (Galaxies, CMB, Clusters))
        if is_compatible_real(a, b):
            assert real(a, b) == real(b, a)
        if is_compatible_harmonic(a, b):
            assert harmonic(a, b) == harmonic(b, a)
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


def test_extract_all_tracers_types_cells_include_maybe(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cells

    assert extract_all_measured_types(
        sacc_data, include_maybe_types=True
    ) == extract_all_measured_types(sacc_data)


def test_extract_all_tracers_types_cwindows_include_maybe(
    sacc_galaxy_cwindows: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_cwindows

    assert extract_all_measured_types(
        sacc_data, include_maybe_types=True
    ) == extract_all_measured_types(sacc_data)


def test_extract_all_tracers_types_xi_thetas_include_maybe(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict],
):
    sacc_data, _, _ = sacc_galaxy_xis

    assert extract_all_measured_types(
        sacc_data, include_maybe_types=True
    ) == extract_all_measured_types(sacc_data)


def test_measurements_from_index1():
    index: TwoPointRealIndex = {
        "data_type": "galaxy_shearDensity_xi_t",
        "tracer_names": TracerNames("src0", "lens0"),
    }
    n1, a, n2, b = measurements_from_index(index)
    assert n1 == "lens0"
    assert a == Galaxies.COUNTS
    assert n2 == "src0"
    assert b == Galaxies.SHEAR_T


def test_measurements_from_index2():
    index: TwoPointRealIndex = {
        "data_type": "galaxy_shearDensity_xi_t",
        "tracer_names": TracerNames("lens0", "src0"),
    }
    n1, a, n2, b = measurements_from_index(index)
    assert n1 == "lens0"
    assert a == Galaxies.COUNTS
    assert n2 == "src0"
    assert b == Galaxies.SHEAR_T


def test_match_name_type1():
    match, n1, a, n2, b = match_name_type(
        "src0", "lens0", Galaxies.SHEAR_T, Galaxies.COUNTS
    )
    assert (
        match
        and n1 == "lens0"
        and a == Galaxies.COUNTS
        and n2 == "src0"
        and b == Galaxies.SHEAR_T
    )


def test_match_name_type2():
    match, n1, a, n2, b = match_name_type(
        "src0", "lens0", Galaxies.COUNTS, Galaxies.SHEAR_T
    )
    assert (
        match
        and n1 == "lens0"
        and a == Galaxies.COUNTS
        and n2 == "src0"
        and b == Galaxies.SHEAR_T
    )


def test_match_name_type3():
    match, n1, a, n2, b = match_name_type(
        "lens0", "src0", Galaxies.SHEAR_T, Galaxies.COUNTS
    )
    assert (
        match
        and n1 == "lens0"
        and a == Galaxies.COUNTS
        and n2 == "src0"
        and b == Galaxies.SHEAR_T
    )


def test_match_name_type4():
    match, n1, a, n2, b = match_name_type(
        "lens0", "src0", Galaxies.COUNTS, Galaxies.SHEAR_T
    )
    assert (
        match
        and n1 == "lens0"
        and a == Galaxies.COUNTS
        and n2 == "src0"
        and b == Galaxies.SHEAR_T
    )


def test_match_name_type_convention1():
    match, n1, a, n2, b = match_name_type(
        "lens0", "no_convention", Galaxies.COUNTS, Galaxies.COUNTS
    )
    assert not match
    assert n1 == "lens0"
    assert a == Galaxies.COUNTS
    assert n2 == "no_convention"
    assert b == Galaxies.COUNTS


def test_match_name_type_convention2():
    match, n1, a, n2, b = match_name_type(
        "no_convention", "lens0", Galaxies.COUNTS, Galaxies.COUNTS
    )
    assert not match
    assert n1 == "no_convention"
    assert a == Galaxies.COUNTS
    assert n2 == "lens0"
    assert b == Galaxies.COUNTS


def test_match_name_type_convention3():
    match, n1, a, n2, b = match_name_type(
        "no_convention", "here_too", Galaxies.COUNTS, Galaxies.SHEAR_T
    )
    assert not match
    assert n1 == "no_convention"
    assert a == Galaxies.COUNTS
    assert n2 == "here_too"
    assert b == Galaxies.SHEAR_T


def test_match_name_type_require_convention_fail():
    with pytest.raises(
        ValueError,
        match="Invalid tracer names (.*) do not respect the naming convention.",
    ):
        match_name_type(
            "no_convention",
            "here_too",
            Galaxies.COUNTS,
            Galaxies.SHEAR_T,
            require_convention=True,
        )


def test_match_name_type_require_convention_lens():
    match, n1, a, n2, b = match_name_type(
        "lens0",
        "lens0",
        Galaxies.COUNTS,
        Galaxies.COUNTS,
        require_convention=True,
    )
    assert not match
    assert n1 == "lens0"
    assert a == Galaxies.COUNTS
    assert n2 == "lens0"
    assert b == Galaxies.COUNTS


def test_match_name_type_require_convention_source():
    match, n1, a, n2, b = match_name_type(
        "src0",
        "src0",
        Galaxies.PART_OF_XI_MINUS,
        Galaxies.PART_OF_XI_MINUS,
        require_convention=True,
    )
    assert not match
    assert n1 == "src0"
    assert a == Galaxies.PART_OF_XI_MINUS
    assert n2 == "src0"
    assert b == Galaxies.PART_OF_XI_MINUS


def test_check_two_point_consistence_harmonic(two_point_cell: TwoPointHarmonic):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=two_point_cell,
    )
    check_two_point_consistence_harmonic([tpm])


def test_check_two_point_consistence_harmonic_real(two_point_real: TwoPointHarmonic):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=two_point_real,
    )
    with pytest.raises(
        ValueError,
        match=(".*is not a measurement of TwoPointHarmonic."),
    ):
        check_two_point_consistence_harmonic([tpm])


def test_check_two_point_consistence_real(two_point_real: TwoPointReal):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=two_point_real,
    )
    check_two_point_consistence_real([tpm])


def test_check_two_point_consistence_real_harmonic(two_point_cell: TwoPointReal):
    tpm = TwoPointMeasurement(
        data=np.zeros(100),
        indices=np.arange(100),
        covariance_name="cov",
        metadata=two_point_cell,
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


def test_extract_all_data_cells_ambiguous(sacc_galaxy_cells_ambiguous: sacc.Sacc):
    with pytest.raises(
        ValueError,
        match=(
            "Ambiguous measurements for tracers .*. Impossible to "
            "determine which measurement is from which tracer."
        ),
    ):
        _ = extract_all_harmonic_data(
            sacc_galaxy_cells_ambiguous, include_maybe_types=True
        )
