"""Functions for converting between measurement types and SACC strings."""

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
