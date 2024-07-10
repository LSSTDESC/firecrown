"""This module deals with two-point types.

This module contains two-point types definitions.
"""

from itertools import chain
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from firecrown.utils import YAMLSerializable


@dataclass(frozen=True)
class TracerNames(YAMLSerializable):
    """The names of the two tracers in the sacc file."""

    name1: str
    name2: str

    def __getitem__(self, item):
        """Get the name of the tracer at the given index."""
        if item == 0:
            return self.name1
        if item == 1:
            return self.name2
        raise IndexError

    def __iter__(self):
        """Iterate through the data members.

        This is to allow automatic unpacking.
        """
        yield self.name1
        yield self.name2


TRACER_NAMES_TOTAL = TracerNames("", "")  # special name to represent total


class Galaxies(YAMLSerializable, str, Enum):
    """This enumeration type for galaxy measurements.

    It provides identifiers for the different types of galaxy-related types of
    measurement.

    SACC has some notion of supporting other types, but incomplete implementation. When
    support for more types is added to SACC this enumeration needs to be updated.
    """

    SHEAR_E = auto()
    SHEAR_T = auto()
    SHEAR_MINUS = auto()
    SHEAR_PLUS = auto()
    COUNTS = auto()

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
        if self == Galaxies.SHEAR_MINUS:
            return "shear"
        if self == Galaxies.SHEAR_PLUS:
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
        if self == Galaxies.SHEAR_MINUS:
            return "minus"
        if self == Galaxies.SHEAR_PLUS:
            return "plus"
        if self == Galaxies.COUNTS:
            return ""
        raise ValueError("Untranslated Galaxy Measurement encountered")

    def __lt__(self, other):
        """Define a comparison function for the Galaxy Measurement enumeration."""
        return compare_enums(self, other) < 0

    def __eq__(self, other):
        """Define an equality test for Galaxy Measurement enumeration."""
        return compare_enums(self, other) == 0

    def __ne__(self, other):
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

    def __lt__(self, other):
        """Define a comparison function for the CMBMeasurement enumeration."""
        return compare_enums(self, other) < 0

    def __eq__(self, other):
        """Define an equality test for CMBMeasurement enumeration."""
        return compare_enums(self, other) == 0

    def __ne__(self, other):
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

    def __lt__(self, other):
        """Define a comparison function for the ClusterMeasurement enumeration."""
        return compare_enums(self, other) < 0

    def __eq__(self, other):
        """Define an equality test for ClusterMeasurement enumeration."""
        return compare_enums(self, other) == 0

    def __ne__(self, other):
        """Negation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Define a hash function that uses both type and value information."""
        return hash((Clusters, self.value))


Measurement = Galaxies | CMB | Clusters
ALL_MEASUREMENTS: list[Measurement] = list(chain(Galaxies, CMB, Clusters))
ALL_MEASUREMENT_TYPES = (Galaxies, CMB, Clusters)
HARMONIC_ONLY_MEASUREMENTS = (Galaxies.SHEAR_E,)
REAL_ONLY_MEASUREMENTS = (Galaxies.SHEAR_T, Galaxies.SHEAR_MINUS, Galaxies.SHEAR_PLUS)
EXACT_MATCH_MEASUREMENTS = (Galaxies.SHEAR_MINUS, Galaxies.SHEAR_PLUS)


def compare_enums(a: Measurement, b: Measurement) -> int:
    """Define a comparison function for the Measurement enumeration.

    Return -1 if a comes before b, 0 if they are the same, and +1 if b comes before a.
    """
    order = (CMB, Clusters, Galaxies)
    main_type_index_a = order.index(type(a))
    main_type_index_b = order.index(type(b))
    if main_type_index_a == main_type_index_b:
        return int(a) - int(b)
    return main_type_index_a - main_type_index_b


@dataclass(frozen=True, kw_only=True)
class InferredGalaxyZDist(YAMLSerializable):
    """The class used to store the redshift resolution data for a sacc file.

    The sacc file is a complicated set of tracers (bins) and surveys. This class is
    used to store the redshift resolution data for a single photometric bin.
    """

    bin_name: str
    z: np.ndarray
    dndz: np.ndarray
    measurements: set[Measurement]

    def __post_init__(self) -> None:
        """Validate the redshift resolution data.

        - Make sure the z and dndz arrays have the same shape;
        - The measurement must be of type Measurement.
        - The bin_name should not be empty.
        """
        if self.z.shape != self.dndz.shape:
            raise ValueError("The z and dndz arrays should have the same shape.")

        for measurement in self.measurements:
            if not isinstance(measurement, ALL_MEASUREMENT_TYPES):
                raise ValueError("The measurement should be a Measurement.")

        if self.bin_name == "":
            raise ValueError("The bin_name should not be empty.")

    def __eq__(self, other):
        """Equality test for InferredGalaxyZDist.

        Two InferredGalaxyZDist are equal if they have equal bin_name, z, dndz, and
        measurement.
        """
        assert isinstance(other, InferredGalaxyZDist)
        return (
            self.bin_name == other.bin_name
            and np.array_equal(self.z, other.z)
            and np.array_equal(self.dndz, other.dndz)
            and self.measurements == other.measurements
        )


@dataclass(frozen=True, kw_only=True)
class TwoPointMeasurement(YAMLSerializable):
    """Class defining the metadata for a two-point measurement.

    The class used to store the metadata for a two-point function measured on a sphere.

    This includes the measured two-point function and their indices in the covariance
    matrix.
    """

    data: npt.NDArray[np.float64]
    indices: npt.NDArray[np.int64]
    covariance_name: str

    def __post_init__(self) -> None:
        """Make sure the data and indices have the same shape."""
        if len(self.data.shape) != 1:
            raise ValueError("Data should be a 1D array.")

        if self.data.shape != self.indices.shape:
            raise ValueError("Data and indices should have the same shape.")

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointMeasurement objects."""
        return (
            np.array_equal(self.data, other.data)
            and np.array_equal(self.indices, other.indices)
            and self.covariance_name == other.covariance_name
        )
