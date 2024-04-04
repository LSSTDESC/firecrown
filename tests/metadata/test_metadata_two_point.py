"""
Tests for the module firecrown.metadata.two_point
"""

from itertools import product

from firecrown.metadata.two_point import (
    GalaxyMeasuredType,
    CMBMeasuredType,
    ClusterMeasuredType,
)
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
        real(GalaxyMeasuredType.SHEAR_T, CMBMeasuredType.CONVERGENCE)
        == "cmbGalaxy_convergenceShear_xi_t"
    )


def test_translation_invariants():
    for a, b in product(ALL_MEASURED_TYPES, ALL_MEASURED_TYPES):
        assert isinstance(a, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        assert isinstance(b, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        assert harmonic(a, b) == harmonic(b, a)
        assert real(a, b) == real(b, a)
        assert harmonic(a, b) != real(a, b)