"""Measurement type definitions and related constants.

This module defines the measurement types (Galaxies, CMB, Clusters) and related
constants used throughout Firecrown.
"""

from __future__ import annotations

import re
from enum import Enum, auto
from itertools import chain

from firecrown.utils import YAMLSerializable


def _compare_enums(a: object, b: object) -> int:
    """Define a comparison function for the Measurement enumeration.

    Return -1 if a comes before b, 0 if they are the same, and +1 if b comes before a.

    This function is defined before the enum classes to avoid circular imports.
    The order checking is deferred until runtime.

    **Ordering Convention for SACC Cross-Correlations:**

    The ordering of Measurement types is critical for interpreting SACC data files.
    When cross-correlating two tracers with different measurement types, the SACC
    convention requires that the measurement type with the lower enum value appears
    first in both:

    1. The SACC data type string (e.g., 'galaxy_shearDensity_cl_e')
    2. The tracer order in data points (e.g., (src0, lens0))

    For example:
        - If src0 has SHEAR_E (enum value 1) and lens0 has COUNTS (enum value 5)
        - SHEAR_E < COUNTS numerically
        - The SACC string must be 'galaxy_shearDensity_cl_e' (shear before density)
        - The tracer order must be (src0, lens0), NOT (lens0, src0)
        - The reverse order (lens0, src0) would violate the SACC convention

    This ordering is enforced by the __lt__ method implementation in each
    Measurement enum class and is validated during SACC file processing.

    :param a: the first value to compare. Typed as `object` (rather than
        `Measurement`) because the Measurement `__lt__`/`__eq__`/`__ne__`
        overrides must accept whatever their `str`/`object` superclass methods
        accept (see the comment above `Galaxies.__lt__` below); the actual
        `Measurement` membership check happens here, at runtime.
    :param b: the second value to compare, see `a` above.
    :raises ValueError: if either `a` or `b` is not one of `CMB`, `Clusters`,
        or `Galaxies`, i.e. not a `Measurement`.
    """
    # Get the order dynamically to avoid forward references
    # We know CMB, Clusters, Galaxies will be defined in this module
    order = (CMB, Clusters, Galaxies)
    if not isinstance(a, order) or not isinstance(b, order):
        raise ValueError(
            f"Unknown measurement type encountered ({type(a)}, {type(b)})."
        )
    # The isinstance check above narrows `a` and `b` from `object` to
    # `CMB | Clusters | Galaxies` (i.e. `Measurement`) for the remainder of
    # this function, so the `int()` conversions below type-check correctly.

    main_type_index_a = order.index(type(a))
    main_type_index_b = order.index(type(b))
    if main_type_index_a == main_type_index_b:
        return int(a) - int(b)
    return main_type_index_a - main_type_index_b


class Galaxies(YAMLSerializable, str, Enum):
    """This enumeration type for galaxy measurements.

    It provides identifiers for the different types of galaxy-related types of
    measurement.

    SACC has some notion of supporting other types, but incomplete implementation. When
    support for more types is added to SACC this enumeration needs to be updated.
    """

    SHEAR_E = auto()
    SHEAR_T = auto()
    PART_OF_XI_MINUS = auto()
    SHEAR_MINUS = PART_OF_XI_MINUS  # Alias for backward compatibility in user code
    PART_OF_XI_PLUS = auto()
    SHEAR_PLUS = PART_OF_XI_PLUS  # Alias for backward compatibility in user code
    COUNTS = auto()

    def is_shear(self) -> bool:
        """Return True if the measurement is a shear measurement, False otherwise.

        :returns: True if the measurement is a shear measurement, False otherwise
        """
        return self in (
            Galaxies.SHEAR_E,
            Galaxies.SHEAR_T,
            Galaxies.PART_OF_XI_MINUS,
            Galaxies.PART_OF_XI_PLUS,
        )

    def sacc_type_name(self) -> str:
        """Return the lower-case form of the main measurement type.

        This is the first part of the SACC string used to denote a correlation between
        measurements of this type.
        """
        return "galaxy"

    def sacc_measurement_name(self) -> str:
        """Return the lower-case form of the specific measurement type.

        This is the second part of the SACC string used to denote the specific
        measurement type.
        """
        if self == Galaxies.SHEAR_E:
            return "shear"
        if self == Galaxies.SHEAR_T:
            return "shear"
        if self == Galaxies.PART_OF_XI_MINUS:
            return "shear"
        if self == Galaxies.PART_OF_XI_PLUS:
            return "shear"
        if self == Galaxies.COUNTS:
            return "density"
        raise ValueError("Untranslated Galaxy Measurement encountered")

    def polarization(self) -> str:
        """Return the SACC polarization code.

        This is the third part of the SACC string used to denote the specific
        measurement type.
        """
        if self == Galaxies.SHEAR_E:
            return "e"
        if self == Galaxies.SHEAR_T:
            return "t"
        if self == Galaxies.PART_OF_XI_MINUS:
            return "minus"
        if self == Galaxies.PART_OF_XI_PLUS:
            return "plus"
        if self == Galaxies.COUNTS:
            return ""
        raise ValueError("Untranslated Galaxy Measurement encountered")

    # Galaxies subclasses both `str` and `Enum`, so mypy checks these overrides
    # against *both* superclasses: `str.__lt__` requires `other: str`, while
    # `str`/`object` `__eq__`/`__ne__` require `other: object`. Narrowing the
    # parameter to `Measurement` (as before) violates the Liskov substitution
    # principle, since an override must accept everything the superclass
    # method accepts. The parameter types below are widened to match the
    # superclasses purely to satisfy that contract; the actual
    # `Measurement`-membership check (and the `ValueError` raised for
    # anything else) still happens inside `_compare_enums`, so runtime
    # behavior for callers is unchanged.
    def __lt__(self, other: str) -> bool:
        """Define a comparison function for the Galaxy Measurement enumeration."""
        return _compare_enums(self, other) < 0

    def __eq__(self, other: object) -> bool:
        """Define an equality test for Galaxy Measurement enumeration."""
        return _compare_enums(self, other) == 0

    def __ne__(self, other: object) -> bool:
        """Negation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Define a hash function that uses both type and value information."""
        return hash((Galaxies, self.value))


class CMB(YAMLSerializable, str, Enum):
    """This enumeration type for CMB measurements.

    It provides identifiers for the different types of CMB-related types of
    measurement.

    SACC has some notion of supporting other types, but incomplete implementation. When
    support for more types is added to SACC this enumeration needs to be updated.
    """

    CONVERGENCE = auto()

    def sacc_type_name(self) -> str:
        """Return the lower-case form of the main measurement type.

        This is the first part of the SACC string used to denote a correlation between
        measurements of this type.
        """
        return "cmb"

    def sacc_measurement_name(self) -> str:
        """Return the lower-case form of the specific measurement type.

        This is the second part of the SACC string used to denote the specific
        measurement type.
        """
        if self == CMB.CONVERGENCE:
            return "convergence"
        raise ValueError("Untranslated CMBMeasurement encountered")

    def polarization(self) -> str:
        """Return the SACC polarization code.

        This is the third part of the SACC string used to denote the specific
        measurement type.
        """
        if self == CMB.CONVERGENCE:
            return ""
        raise ValueError("Untranslated CMBMeasurement encountered")

    # See the comment on Galaxies.__lt__ above: CMB also subclasses both `str`
    # and `Enum`, so the same widened parameter types are needed to satisfy
    # the Liskov substitution principle against both `str` and `object`. The
    # `Measurement`-membership check (and its `ValueError`) still happens
    # inside `_compare_enums`, so runtime behavior is unchanged.
    def __lt__(self, other: str) -> bool:
        """Define a comparison function for the CMBMeasurement enumeration."""
        return _compare_enums(self, other) < 0

    def __eq__(self, other: object) -> bool:
        """Define an equality test for CMBMeasurement enumeration."""
        return _compare_enums(self, other) == 0

    def __ne__(self, other: object) -> bool:
        """Negation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Define a hash function that uses both type and value information."""
        return hash((CMB, self.value))


class Clusters(YAMLSerializable, str, Enum):
    """This enumeration type for cluster measurements.

    It provides identifiers for the different types of cluster-related types of
    measurement.

    SACC has some notion of supporting other types, but incomplete implementation. When
    support for more types is added to SACC this enumeration needs to be updated.
    """

    COUNTS = auto()

    def sacc_type_name(self) -> str:
        """Return the lower-case form of the main measurement type.

        This is the first part of the SACC string used to denote a correlation between
        measurements of this type.
        """
        return "cluster"

    def sacc_measurement_name(self) -> str:
        """Return the lower-case form of the specific measurement type.

        This is the second part of the SACC string used to denote the specific
        measurement type.
        """
        if self == Clusters.COUNTS:
            return "density"
        raise ValueError("Untranslated ClusterMeasurement encountered")

    def polarization(self) -> str:
        """Return the SACC polarization code.

        This is the third part of the SACC string used to denote the specific
        measurement type.
        """
        if self == Clusters.COUNTS:
            return ""
        raise ValueError("Untranslated ClusterMeasurement encountered")

    # See the comment on Galaxies.__lt__ above: Clusters also subclasses both
    # `str` and `Enum`, so the same widened parameter types are needed to
    # satisfy the Liskov substitution principle against both `str` and
    # `object`. The `Measurement`-membership check (and its `ValueError`)
    # still happens inside `_compare_enums`, so runtime behavior is
    # unchanged.
    def __lt__(self, other: str) -> bool:
        """Define a comparison function for the ClusterMeasurement enumeration."""
        return _compare_enums(self, other) < 0

    def __eq__(self, other: object) -> bool:
        """Define an equality test for ClusterMeasurement enumeration."""
        return _compare_enums(self, other) == 0

    def __ne__(self, other: object) -> bool:
        """Negation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Define a hash function that uses both type and value information."""
        return hash((Clusters, self.value))


Measurement = Galaxies | CMB | Clusters
"""Type alias for any measurement type."""

ALL_MEASUREMENTS: list[Measurement] = list(chain(Galaxies, CMB, Clusters))
"""Comprehensive list of all measurement types.

Includes all measurement types across all categories (Galaxies, CMB, Clusters).
"""

ALL_MEASUREMENT_TYPES = (Galaxies, CMB, Clusters)
"""Tuple of all measurement type classes.

Used for type checking and iteration over measurement categories.
"""

HARMONIC_ONLY_MEASUREMENTS = (Galaxies.SHEAR_E,)
"""Galaxy measurements analyzable only in harmonic space.

Currently includes shear E-mode measurements, which can only be analyzed in
Fourier/harmonic space.
"""

REAL_ONLY_MEASUREMENTS = (
    Galaxies.SHEAR_T,
    Galaxies.PART_OF_XI_MINUS,
    Galaxies.PART_OF_XI_PLUS,
)
"""Galaxy measurements analyzable only in real space.

Includes shear T-mode and correlation function components (xi_minus, xi_plus), which
can only be analyzed in configuration/real space.
"""

EXACT_MATCH_MEASUREMENTS = (Galaxies.PART_OF_XI_MINUS, Galaxies.PART_OF_XI_PLUS)
"""Measurements requiring exact type matching in correlations.

Used for xi_minus and xi_plus correlation functions, where both measurements in a
correlation must be of the exact same type.
"""

INCOMPATIBLE_MEASUREMENTS = (Galaxies.SHEAR_T,)
"""Measurements incompatible with cross-correlations.

Currently includes shear T-mode measurements, which cannot be correlated with other
measurement types.
"""

LENS_REGEX = re.compile(r"^lens\d+$")
"""Regular expression for lens tracer bin names.

Matches names like lens0, lens1, lens2, etc.
"""

SOURCE_REGEX = re.compile(r"^(src\d+|source\d+)$")
"""Regular expression for source tracer bin names.

Matches names like src0, src1, source0, source1, etc.
"""

GALAXY_SOURCE_TYPES = (
    Galaxies.SHEAR_E,
    Galaxies.SHEAR_T,
    Galaxies.PART_OF_XI_MINUS,
    Galaxies.PART_OF_XI_PLUS,
)
"""Galaxy measurements for source tracers.

Includes all shear-related measurements associated with source (weak lensing)
tracers.
"""

GALAXY_LENS_TYPES = (Galaxies.COUNTS,)
"""Galaxy measurements for lens tracers.

Includes galaxy number counts (density) associated with lens (clustering) tracers.
"""

CMB_TYPES = tuple(CMB)
"""Tuple of all CMB measurement types."""

CLUSTER_TYPES = tuple(Clusters)
"""Tuple of all cluster measurement types."""
