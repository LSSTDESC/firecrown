"""Functions for checking measurement compatibility."""

from firecrown.metadata_types._measurements import (
    EXACT_MATCH_MEASUREMENTS,
    HARMONIC_ONLY_MEASUREMENTS,
    INCOMPATIBLE_MEASUREMENTS,
    REAL_ONLY_MEASUREMENTS,
    Measurement,
)


def measurement_is_compatible(a: Measurement, b: Measurement) -> bool:
    """Check if two Measurement are compatible.

    Two Measurement are compatible if they can be correlated in a two-point function.
    """
    if a in HARMONIC_ONLY_MEASUREMENTS and b in REAL_ONLY_MEASUREMENTS:
        return False
    if a in REAL_ONLY_MEASUREMENTS and b in HARMONIC_ONLY_MEASUREMENTS:
        return False
    if (a in EXACT_MATCH_MEASUREMENTS or b in EXACT_MATCH_MEASUREMENTS) and a != b:
        return False
    if a in INCOMPATIBLE_MEASUREMENTS and b in INCOMPATIBLE_MEASUREMENTS:
        return False
    # This enforces the SACC convention on the ordering of measurements. The ordering
    # should follow the canonical SACC ordering: CMB < Clusters < Galaxies, which is
    # implemented in compare_enums. Within each type, the ordering is defined by the
    # Measurement enum order. For example, for Galaxies, the ordering is: shape
    # measurements (shear, usually named src or source) followed by counts measurements
    # (usually named lens).
    if b < a:
        return False
    return True


def _measurement_supports_real(x: Measurement) -> bool:
    """Return True if x supports real-space calculations."""
    return x not in HARMONIC_ONLY_MEASUREMENTS


def _measurement_supports_harmonic(x: Measurement) -> bool:
    """Return True if x supports harmonic-space calculations."""
    return x not in REAL_ONLY_MEASUREMENTS


def _measurement_is_compatible_real(a: Measurement, b: Measurement) -> bool:
    """Check if two Measurement are compatible for real-space calculations.

    Two Measurement are compatible if they can be correlated in a real-space two-point
    function.
    """
    return (
        _measurement_supports_real(a)
        and _measurement_supports_real(b)
        and measurement_is_compatible(a, b)
    )


def _measurement_is_compatible_harmonic(a: Measurement, b: Measurement) -> bool:
    """Check if two Measurement are compatible for harmonic-space calculations.

    Two Measurement are compatible if they can be correlated in a harmonic-space
    two-point function.
    """
    return (
        _measurement_supports_harmonic(a)
        and _measurement_supports_harmonic(b)
        and measurement_is_compatible(a, b)
    )


def measurements_types(
    measurements: set[Measurement],
) -> tuple[bool, list[str]]:
    """Collect the types of the measurements in a set.

    Return a tuple (bool, list[str]) where the first element is True if the set of
    measurements contains more than one type of measurement, and the second element is
    a list of the types of the measurements.
    """
    list_types = []
    if measurements & set(GALAXY_LENS_TYPES):
        list_types.append("Galaxy lens")
    if measurements & set(GALAXY_SOURCE_TYPES):
        list_types.append("Galaxy source")
    if measurements & set(CMB_TYPES):
        list_types.append("CMB")
    if measurements & set(CLUSTER_TYPES):
        list_types.append("Clusters")

    return len(list_types) > 1, list_types
