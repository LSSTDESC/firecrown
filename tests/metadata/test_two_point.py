"""
Tests for the module firecrown.metadata.two_point
"""

from firecrown.metadata.two_point import GalaxyMeasuredType, CMBMeasuredType
from firecrown.metadata.two_point import (
    ALL_MEASURED_TYPES,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
)


def test_exact_matches():
    assert (
        harmonic(GalaxyMeasuredType.SHEAR_E, CMBMeasuredType.CONVERGENCE)
        == "cmbGalaxy_convergenceShear_cl_e"
    )
    assert (
        real(GalaxyMeasuredType.SHEAR_E, CMBMeasuredType.CONVERGENCE)
        == "cmbGalaxy_convergenceShear_xi_t"
    )


def test_translation_invariants():
    for a in ALL_MEASURED_TYPES:
        for b in ALL_MEASURED_TYPES:
            assert harmonic(a, b) == harmonic(b, a)
            assert real(a, b) == real(b, a)
            assert harmonic(a, b) != real(a, b)
