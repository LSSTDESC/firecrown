"""
Tests for the module firecrown.metadata.two_point
"""

from itertools import product, chain
from unittest.mock import MagicMock
import re
import pytest


import sacc
import sacc_name_mapping as snm
from firecrown.metadata.two_point_types import (
    Galaxies,
    Clusters,
    CMB,
    ALL_MEASUREMENTS,
    compare_enums,
)


from firecrown.metadata.two_point import (
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
    extract_all_tracers_types,
    measurement_is_compatible as is_compatible,
    measurement_is_compatible_real as is_compatible_real,
    measurement_is_compatible_harmonic as is_compatible_harmonic,
    measurement_supports_harmonic as supports_harmonic,
    measurement_supports_real as supports_real,
    LENS_REGEX,
    SOURCE_REGEX,
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
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict]
):
    sacc_data, _, _ = sacc_galaxy_cells

    tracers = extract_all_tracers_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.COUNTS
        if SOURCE_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.SHEAR_E


def test_extract_all_tracers_types_cwindows(
    sacc_galaxy_cwindows: tuple[sacc.Sacc, dict, dict]
):
    sacc_data, _, _ = sacc_galaxy_cwindows

    tracers = extract_all_tracers_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.COUNTS
        if SOURCE_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.SHEAR_E


def test_extract_all_tracers_types_xi_thetas(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict]
):
    sacc_data, _, _ = sacc_galaxy_xis

    tracers = extract_all_tracers_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.COUNTS
        if SOURCE_REGEX.match(tracer):
            assert measurements == {
                Galaxies.SHEAR_T,
                Galaxies.SHEAR_MINUS,
                Galaxies.SHEAR_PLUS,
            }


def test_extract_all_tracers_types_xi_thetas_inverted(
    sacc_galaxy_xis_inverted: tuple[sacc.Sacc, dict, dict]
):
    sacc_data, _, _ = sacc_galaxy_xis_inverted

    tracers = extract_all_tracers_types(sacc_data)

    for tracer, measurements in tracers.items():
        if LENS_REGEX.match(tracer):
            for measurement in measurements:
                assert measurement == Galaxies.COUNTS
        if SOURCE_REGEX.match(tracer):
            assert measurements == {
                Galaxies.SHEAR_T,
                Galaxies.SHEAR_MINUS,
                Galaxies.SHEAR_PLUS,
            }


def test_extract_all_tracers_types_cells_include_maybe(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict, dict]
):
    sacc_data, _, _ = sacc_galaxy_cells

    assert extract_all_tracers_types(
        sacc_data, include_maybe_types=True
    ) == extract_all_tracers_types(sacc_data)


def test_extract_all_tracers_types_cwindows_include_maybe(
    sacc_galaxy_cwindows: tuple[sacc.Sacc, dict, dict]
):
    sacc_data, _, _ = sacc_galaxy_cwindows

    assert extract_all_tracers_types(
        sacc_data, include_maybe_types=True
    ) == extract_all_tracers_types(sacc_data)


def test_extract_all_tracers_types_xi_thetas_include_maybe(
    sacc_galaxy_xis: tuple[sacc.Sacc, dict, dict]
):
    sacc_data, _, _ = sacc_galaxy_xis

    assert extract_all_tracers_types(
        sacc_data, include_maybe_types=True
    ) == extract_all_tracers_types(sacc_data)
