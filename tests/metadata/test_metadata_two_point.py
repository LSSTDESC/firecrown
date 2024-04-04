"""
Tests for the module firecrown.metadata.two_point
"""

from itertools import product, chain
import sacc_name_mapping as snm
from firecrown.metadata.two_point import (
    GalaxyMeasuredType,
    CMBMeasuredType,
    ClusterMeasuredType,
    compare_enums,
    ALL_MEASURED_TYPES,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
    measured_type_supports_harmonic as supports_harmonic,
    measured_type_supports_real as supports_real,
)


def test_order_enums():
    assert compare_enums(CMBMeasuredType.CONVERGENCE, ClusterMeasuredType.COUNTS) < 0
    assert compare_enums(ClusterMeasuredType.COUNTS, CMBMeasuredType.CONVERGENCE) > 0

    assert compare_enums(CMBMeasuredType.CONVERGENCE, GalaxyMeasuredType.COUNTS) < 0
    assert compare_enums(GalaxyMeasuredType.COUNTS, CMBMeasuredType.CONVERGENCE) > 0

    assert compare_enums(GalaxyMeasuredType.SHEAR_E, GalaxyMeasuredType.SHEAR_T) < 0
    assert compare_enums(GalaxyMeasuredType.SHEAR_E, GalaxyMeasuredType.COUNTS) < 0
    assert compare_enums(GalaxyMeasuredType.SHEAR_T, GalaxyMeasuredType.COUNTS) < 0

    assert compare_enums(GalaxyMeasuredType.COUNTS, GalaxyMeasuredType.SHEAR_E) > 0

    for enumerand in ALL_MEASURED_TYPES:
        assert compare_enums(enumerand, enumerand) == 0


def test_enumeration_equality_galaxy():
    for e1, e2 in product(
        GalaxyMeasuredType, chain(CMBMeasuredType, ClusterMeasuredType)
    ):
        assert e1 != e2


def test_enumeration_equality_cmb():
    for e1, e2 in product(
        CMBMeasuredType, chain(GalaxyMeasuredType, ClusterMeasuredType)
    ):
        assert e1 != e2


def test_enumeration_equality_cluster():
    for e1, e2 in product(
        ClusterMeasuredType, chain(CMBMeasuredType, GalaxyMeasuredType)
    ):
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
    for a, b in product(ALL_MEASURED_TYPES, ALL_MEASURED_TYPES):
        assert isinstance(a, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        assert isinstance(b, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        if supports_real(a) and supports_real(b):
            assert real(a, b) == real(b, a)
        if supports_harmonic(a) and supports_harmonic(b):
            assert harmonic(a, b) == harmonic(b, a)
        if (
            supports_harmonic(a)
            and supports_harmonic(b)
            and supports_real(a)
            and supports_real(b)
        ):
            assert harmonic(a, b) != real(a, b)
