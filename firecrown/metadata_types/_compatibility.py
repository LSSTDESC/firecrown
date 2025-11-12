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
