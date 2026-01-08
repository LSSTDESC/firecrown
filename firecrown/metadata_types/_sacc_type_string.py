"""Functions for converting between measurement types and SACC strings.

SACC Tracer Ordering Convention
================================

When SACC stores a cross-correlation between two tracers with different measurement
types, the ordering of tracers must match the ordering of measurement types in the
data type string. This convention ensures unambiguous interpretation of two-point
measurements.

The ordering is based on the numeric values of the Measurement enums:
    - CMB comes before Clusters comes before Galaxies (by class)
    - Within each class, measurements are ordered by their enum values

**Auto-correlations (same measurement type):**
    - Example: src0 × src0 with SHEAR_E × SHEAR_E → 'galaxy_shear_cl_ee'
    - Both tracers have SHEAR_E, so tracer order doesn't matter
    - SACC string: 'galaxy_shear_cl_ee' (single measurement type)

**Cross-correlations (different measurement types):**
    - Example: src0 has SHEAR_E, lens0 has COUNTS
    - Since SHEAR_E < COUNTS (enum ordering), SHEAR_E must come first
    - SACC string: 'galaxy_shearDensity_cl_e' (shear before density)
    - Tracer order: (src0, lens0) ✓ CORRECT
    - Tracer order: (lens0, src0) ✗ VIOLATION

**How to detect violations:**
    1. From auto-correlations, determine each tracer's measurement type
       - src0 × src0 with 'galaxy_shear_cl_ee' → src0 is SHEAR_E
       - lens0 × lens0 with 'galaxy_density_cl' → lens0 is COUNTS
    2. For cross-correlations, verify tracer order matches type order
       - 'galaxy_shearDensity_cl_e' maps to (SHEAR_E, COUNTS)
       - If data has (lens0, src0) but lens0 is COUNTS and src0 is SHEAR_E,
         this is a violation because COUNTS > SHEAR_E
       - Correct order should be (src0, lens0)

See Also
--------
- extract_all_measured_types() for type extraction logic
- Transform._fix_ordering() for automated correction
- MEASURED_TYPE_STRING_MAP for type string → measurement mapping
"""

from itertools import product

from firecrown.metadata_types._compatibility import (
    measurement_is_compatible_harmonic,
    measurement_is_compatible_real,
)
from firecrown.metadata_types._measurements import (
    ALL_MEASUREMENTS,
    EXACT_MATCH_MEASUREMENTS,
    HARMONIC_ONLY_MEASUREMENTS,
    REAL_ONLY_MEASUREMENTS,
    Measurement,
)


def _type_to_sacc_string_common(x: Measurement, y: Measurement) -> str:
    """Return the first two parts of the SACC string.

    The first two parts of the SACC string is used to denote a correlation between
    measurements of x and y.
    """
    if isinstance(x, type(y)):
        part_1 = f"{x.sacc_type_name()}_"
        if x == y:
            part_2 = f"{x.sacc_measurement_name()}_"
        else:
            part_2 = (
                f"{x.sacc_measurement_name()}{y.sacc_measurement_name().capitalize()}_"
            )
    else:
        part_1 = f"{x.sacc_type_name()}{y.sacc_type_name().capitalize()}_"
        if x.sacc_measurement_name() == y.sacc_measurement_name():
            part_2 = f"{x.sacc_measurement_name()}_"
        else:
            part_2 = (
                f"{x.sacc_measurement_name()}{y.sacc_measurement_name().capitalize()}_"
            )

    return part_1 + part_2


def _type_to_sacc_string_real(x: Measurement, y: Measurement) -> str:
    """Return the final SACC string used to denote the real-space correlation.

    The SACC string used to denote the real-space correlation type between measurements
    of x and y.
    """
    if x in EXACT_MATCH_MEASUREMENTS:
        assert x == y
        suffix = f"{x.polarization()}"
    else:
        suffix = f"{x.polarization()}{y.polarization()}"

    if x in HARMONIC_ONLY_MEASUREMENTS or y in HARMONIC_ONLY_MEASUREMENTS:
        raise ValueError("Real-space correlation not supported for shear E.")

    return _type_to_sacc_string_common(x, y) + (f"xi_{suffix}" if suffix else "xi")


def _type_to_sacc_string_harmonic(x: Measurement, y: Measurement) -> str:
    """Return the final SACC string used to denote the harmonic-space correlation.

    the SACC string used to denote the harmonic-space correlation type between
    measurements of x and y.
    """
    suffix = f"{x.polarization()}{y.polarization()}"

    if x in REAL_ONLY_MEASUREMENTS or y in REAL_ONLY_MEASUREMENTS:
        raise ValueError("Harmonic-space correlation not supported for shear T.")

    return _type_to_sacc_string_common(x, y) + (f"cl_{suffix}" if suffix else "cl")


# Map of SACC string to measurement type pairs
MEASURED_TYPE_STRING_MAP: dict[str, tuple[Measurement, Measurement]] = {
    _type_to_sacc_string_real(a, b): (a, b)
    for a, b in product(ALL_MEASUREMENTS, repeat=2)
    if measurement_is_compatible_real(a, b)
} | {
    _type_to_sacc_string_harmonic(a, b): (a, b)
    for a, b in product(ALL_MEASUREMENTS, repeat=2)
    if measurement_is_compatible_harmonic(a, b)
}
