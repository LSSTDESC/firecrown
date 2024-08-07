"""This module deals with two-point types.

This module contains two-point types definitions.
"""

from itertools import chain, combinations_with_replacement
from dataclasses import dataclass
import re
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from firecrown.utils import compare_optional_arrays, compare_optionals, YAMLSerializable


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
LENS_REGEX = re.compile(r"^lens\d+$")
SOURCE_REGEX = re.compile(r"^(src\d+|source\d+)$")
GALAXY_SOURCE_TYPES = (
    Galaxies.SHEAR_E,
    Galaxies.SHEAR_T,
    Galaxies.SHEAR_MINUS,
    Galaxies.SHEAR_PLUS,
)
GALAXY_LENS_TYPES = (Galaxies.COUNTS,)


def compare_enums(a: Measurement, b: Measurement) -> int:
    """Define a comparison function for the Measurement enumeration.

    Return -1 if a comes before b, 0 if they are the same, and +1 if b comes before a.
    """
    order = (CMB, Clusters, Galaxies)
    if type(a) not in order or type(b) not in order:
        raise ValueError(
            f"Unknown measurement type encountered ({type(a)}, {type(b)})."
        )

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


def make_measurements_dict(value: set[Measurement]) -> list[dict[str, str]]:
    """Create a dictionary from a Measurement object.

    :param value: the measurement to turn into a dictionary
    """
    return [
        {"subject": type(measurement).__name__, "property": measurement.name}
        for measurement in value
    ]


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
    return True


def measurement_supports_real(x: Measurement) -> bool:
    """Return True if x supports real-space calculations."""
    return x not in HARMONIC_ONLY_MEASUREMENTS


def measurement_supports_harmonic(x: Measurement) -> bool:
    """Return True if x supports harmonic-space calculations."""
    return x not in REAL_ONLY_MEASUREMENTS


def measurement_is_compatible_real(a: Measurement, b: Measurement) -> bool:
    """Check if two Measurement are compatible for real-space calculations.

    Two Measurement are compatible if they can be correlated in a real-space two-point
    function.
    """
    return (
        measurement_supports_real(a)
        and measurement_supports_real(b)
        and measurement_is_compatible(a, b)
    )


def measurement_is_compatible_harmonic(a: Measurement, b: Measurement) -> bool:
    """Check if two Measurement are compatible for harmonic-space calculations.

    Two Measurement are compatible if they can be correlated in a harmonic-space
    two-point function.
    """
    return (
        measurement_supports_harmonic(a)
        and measurement_supports_harmonic(b)
        and measurement_is_compatible(a, b)
    )


def _type_to_sacc_string_common(x: Measurement, y: Measurement) -> str:
    """Return the first two parts of the SACC string.

    The first two parts of the SACC string is used to denote a correlation between
    measurements of x and y.
    """
    a, b = sorted([x, y])
    if isinstance(a, type(b)):
        part_1 = f"{a.sacc_type_name()}_"
        if a == b:
            part_2 = f"{a.sacc_measurement_name()}_"
        else:
            part_2 = (
                f"{a.sacc_measurement_name()}{b.sacc_measurement_name().capitalize()}_"
            )
    else:
        part_1 = f"{a.sacc_type_name()}{b.sacc_type_name().capitalize()}_"
        if a.sacc_measurement_name() == b.sacc_measurement_name():
            part_2 = f"{a.sacc_measurement_name()}_"
        else:
            part_2 = (
                f"{a.sacc_measurement_name()}{b.sacc_measurement_name().capitalize()}_"
            )

    return part_1 + part_2


def type_to_sacc_string_real(x: Measurement, y: Measurement) -> str:
    """Return the final SACC string used to denote the real-space correlation.

    The SACC string used to denote the real-space correlation type
    between measurements of x and y.
    """
    a, b = sorted([x, y])
    if a in EXACT_MATCH_MEASUREMENTS:
        assert a == b
        suffix = f"{a.polarization()}"
    else:
        suffix = f"{a.polarization()}{b.polarization()}"

    if a in HARMONIC_ONLY_MEASUREMENTS or b in HARMONIC_ONLY_MEASUREMENTS:
        raise ValueError("Real-space correlation not supported for shear E.")

    return _type_to_sacc_string_common(x, y) + (f"xi_{suffix}" if suffix else "xi")


def type_to_sacc_string_harmonic(x: Measurement, y: Measurement) -> str:
    """Return the final SACC string used to denote the harmonic-space correlation.

    the SACC string used to denote the harmonic-space correlation type
    between measurements of x and y.
    """
    a, b = sorted([x, y])
    suffix = f"{a.polarization()}{b.polarization()}"

    if a in REAL_ONLY_MEASUREMENTS or b in REAL_ONLY_MEASUREMENTS:
        raise ValueError("Harmonic-space correlation not supported for shear T.")

    return _type_to_sacc_string_common(x, y) + (f"cl_{suffix}" if suffix else "cl")


MEASURED_TYPE_STRING_MAP: dict[str, tuple[Measurement, Measurement]] = {
    type_to_sacc_string_real(a, b): (a, b) if a < b else (b, a)
    for a, b in combinations_with_replacement(ALL_MEASUREMENTS, 2)
    if measurement_is_compatible_real(a, b)
} | {
    type_to_sacc_string_harmonic(a, b): (a, b) if a < b else (b, a)
    for a, b in combinations_with_replacement(ALL_MEASUREMENTS, 2)
    if measurement_is_compatible_harmonic(a, b)
}


@dataclass(frozen=True, kw_only=True)
class TwoPointXY(YAMLSerializable):
    """Class defining a two-point correlation pair of redshift resolutions.

    It is used to store the two redshift resolutions for the two bins being
    correlated.
    """

    x: InferredGalaxyZDist
    y: InferredGalaxyZDist
    x_measurement: Measurement
    y_measurement: Measurement

    def __post_init__(self) -> None:
        """Make sure the two redshift resolutions are compatible."""
        if self.x_measurement not in self.x.measurements:
            raise ValueError(
                f"Measurement {self.x_measurement} not in the measurements of "
                f"{self.x.bin_name}."
            )
        if self.y_measurement not in self.y.measurements:
            raise ValueError(
                f"Measurement {self.y_measurement} not in the measurements of "
                f"{self.y.bin_name}."
            )
        if not measurement_is_compatible(self.x_measurement, self.y_measurement):
            raise ValueError(
                f"Measurements {self.x_measurement} and {self.y_measurement} "
                f"are not compatible."
            )

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointXY objects."""
        return (
            self.x == other.x
            and self.y == other.y
            and self.x_measurement == other.x_measurement
            and self.y_measurement == other.y_measurement
        )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointXY object."""
        return f"({self.x.bin_name}, {self.y.bin_name})"

    def get_tracer_names(self) -> TracerNames:
        """Return the TracerNames object for the TwoPointXY object."""
        return TracerNames(self.x.bin_name, self.y.bin_name)


@dataclass(frozen=True, kw_only=True)
class TwoPointCells(YAMLSerializable):
    """Class defining the metadata for an harmonic-space two-point measurement.

    The class used to store the metadata for a (spherical) harmonic-space two-point
    function measured on a sphere.

    This includes the two redshift resolutions (one for each binned quantity) and the
    array of (integer) l's at which the two-point function which has this metadata were
    calculated.
    """

    XY: TwoPointXY
    ells: npt.NDArray[np.int64]
    Cell: None | TwoPointMeasurement = None

    def __post_init__(self) -> None:
        """Validate the TwoPointCells data.

        Make sure the ells are a 1D array and X and Y are compatible
        with harmonic-space calculations.
        """
        if len(self.ells.shape) != 1:
            raise ValueError("Ells should be a 1D array.")

        if self.Cell is not None and self.Cell.data.shape != self.ells.shape:
            raise ValueError("Cell should have the same shape as ells.")

        if not measurement_supports_harmonic(
            self.XY.x_measurement
        ) or not measurement_supports_harmonic(self.XY.y_measurement):
            raise ValueError(
                f"Measurements {self.XY.x_measurement} and "
                f"{self.XY.y_measurement} must support harmonic-space calculations."
            )

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointCells objects."""
        return (
            self.XY == other.XY
            and np.array_equal(self.ells, other.ells)
            and compare_optionals(self.Cell, other.Cell)
        )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointCells object."""
        return f"{self.XY}[{self.get_sacc_name()}]"

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x_measurement, self.XY.y_measurement
        )

    def has_data(self) -> bool:
        """Return True if the TwoPointCells object has a Cell array."""
        return self.Cell is not None


@dataclass(kw_only=True)
class Window(YAMLSerializable):
    """The class used to represent a window function.

    It contains the ells at which the window function is defined, the weights
    of the window function, and the ells at which the window function is
    interpolated.

    It may contain the ells for interpolation if the theory prediction is
    calculated at a different set of ells than the window function.
    """

    ells: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]
    ells_for_interpolation: None | npt.NDArray[np.int64] = None

    def __post_init__(self) -> None:
        """Make sure the weights have the right shape."""
        if len(self.ells.shape) != 1:
            raise ValueError("Ells should be a 1D array.")
        if len(self.weights.shape) != 2:
            raise ValueError("Weights should be a 2D array.")
        if self.weights.shape[0] != len(self.ells):
            raise ValueError("Weights should have the same number of rows as ells.")
        if (
            self.ells_for_interpolation is not None
            and len(self.ells_for_interpolation.shape) != 1
        ):
            raise ValueError("Ells for interpolation should be a 1D array.")

    def n_observations(self) -> int:
        """Return the number of observations supported by the window function."""
        return self.weights.shape[1]

    def __eq__(self, other) -> bool:
        """Equality test for Window objects."""
        assert isinstance(other, Window)
        # We will need special handling for the optional ells_for_interpolation.
        # First handle the non-optinal parts.
        partial_result = np.array_equal(self.ells, other.ells) and np.array_equal(
            self.weights, other.weights
        )
        if not partial_result:
            return False
        return compare_optional_arrays(
            self.ells_for_interpolation, other.ells_for_interpolation
        )


@dataclass(frozen=True, kw_only=True)
class TwoPointCWindow(YAMLSerializable):
    """Two-point function with a window function.

    The class used to store the metadata for a (spherical) harmonic-space two-point
    function measured on a sphere, with an associated window function.

    This includes the two redshift resolutions (one for each binned quantity) and the
    matrix (window function) that relates the measured Cl's with the predicted Cl's.

    Note that the matrix `window` always has l=0 and l=1 suppressed.
    """

    XY: TwoPointXY
    window: Window
    Cell: None | TwoPointMeasurement = None

    def __post_init__(self):
        """Validate the TwoPointCWindow data.

        Make sure the window is
        """
        if not isinstance(self.window, Window):
            raise ValueError("Window should be a Window object.")

        if self.Cell is not None:
            if len(self.Cell.data) != self.window.n_observations():
                raise ValueError(
                    "Data should have the same number of elements as the number of "
                    "observations supported by the window function."
                )

        if not measurement_supports_harmonic(
            self.XY.x_measurement
        ) or not measurement_supports_harmonic(self.XY.y_measurement):
            raise ValueError(
                f"Measurements {self.XY.x_measurement} and "
                f"{self.XY.y_measurement} must support harmonic-space calculations."
            )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointCWindow object."""
        return f"{self.XY}[{self.get_sacc_name()}]"

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x_measurement, self.XY.y_measurement
        )

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointCWindow objects."""
        return (
            self.XY == other.XY
            and self.window == other.window
            and compare_optionals(self.Cell, other.Cell)
        )

    def has_data(self) -> bool:
        """Return True if the TwoPointCWindow object has a Cell array."""
        return self.Cell is not None


@dataclass(frozen=True, kw_only=True)
class TwoPointXiTheta(YAMLSerializable):
    """Class defining the metadata for a real-space two-point measurement.

    The class used to store the metadata for a real-space two-point function measured
    on a sphere.

    This includes the two redshift resolutions (one for each binned quantity) and the a
    array of (floating point) theta (angle) values at which the two-point function
    which has  this metadata were calculated.
    """

    XY: TwoPointXY
    thetas: npt.NDArray[np.float64]
    xis: None | TwoPointMeasurement = None

    def __post_init__(self):
        """Validate the TwoPointCWindow data.

        Make sure the window is
        """
        if len(self.thetas.shape) != 1:
            raise ValueError("Thetas should be a 1D array.")

        if self.xis is not None and self.xis.data.shape != self.thetas.shape:
            raise ValueError("Xis should have the same shape as thetas.")

        if not measurement_supports_real(
            self.XY.x_measurement
        ) or not measurement_supports_real(self.XY.y_measurement):
            raise ValueError(
                f"Measurements {self.XY.x_measurement} and "
                f"{self.XY.y_measurement} must support real-space calculations."
            )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointXiTheta object."""
        return f"{self.XY}[{self.get_sacc_name()}]"

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_real(self.XY.x_measurement, self.XY.y_measurement)

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointXiTheta objects."""
        return (
            self.XY == other.XY
            and np.array_equal(self.thetas, other.thetas)
            and compare_optionals(self.xis, other.xis)
        )

    def has_data(self) -> bool:
        """Return True if the TwoPointXiTheta object has a xis array."""
        return self.xis is not None
