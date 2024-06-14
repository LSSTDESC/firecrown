"""This module deals with two-point functions metadata.

It contains all data classes and functions for store and extract two-point functions
metadata from a sacc file.
"""

from itertools import combinations_with_replacement, chain
from typing import TypedDict, TypeVar, Type
from dataclasses import dataclass
from enum import Enum, auto
import re

import numpy as np
import numpy.typing as npt

import yaml
from yaml import CLoader as Loader
from yaml import CDumper as Dumper

import sacc
from sacc.data_types import required_tags

from firecrown.utils import compare_optional_arrays

ST = TypeVar("ST")  # This will be used in YAMLSerializable


class YAMLSerializable:
    """Protocol for classes that can be serialized to and from YAML."""

    def to_yaml(self: ST) -> str:
        """Return the YAML representation of the object."""
        return yaml.dump(self, Dumper=Dumper, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[ST], yaml_str: str) -> ST:
        """Load the object from YAML."""
        return yaml.load(yaml_str, Loader=Loader)


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
REAL_ONLY_MEASUREMENTS = (Galaxies.SHEAR_T,)


def measurement_is_compatible(a: Measurement, b: Measurement) -> bool:
    """Check if two Measurement are compatible.

    Two Measurement are compatible if they can be correlated in a two-point function.
    """
    if a in HARMONIC_ONLY_MEASUREMENTS and b in REAL_ONLY_MEASUREMENTS:
        return False
    if a in REAL_ONLY_MEASUREMENTS and b in HARMONIC_ONLY_MEASUREMENTS:
        return False
    return True


def measurement_supports_real(x: Measurement) -> bool:
    """Return True if x supports real-space calculations."""
    return x not in HARMONIC_ONLY_MEASUREMENTS


def measurement_supports_harmonic(x: Measurement) -> bool:
    """Return True if x supports harmonic-space calculations."""
    return x not in REAL_ONLY_MEASUREMENTS


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
    measurement: Measurement

    def __post_init__(self) -> None:
        """Validate the redshift resolution data.

        - Make sure the z and dndz arrays have the same shape;
        - The measurement must be of type Measurement.
        - The bin_name should not be empty.
        """
        if self.z.shape != self.dndz.shape:
            raise ValueError("The z and dndz arrays should have the same shape.")

        if not isinstance(self.measurement, ALL_MEASUREMENT_TYPES):
            raise ValueError("The measurement should be a Measurement.")

        if self.bin_name == "":
            raise ValueError("The bin_name should not be empty.")

    def __eq__(self, other):
        """Equality test for InferredGalaxyZDist.

        Two InferredGalaxyZDist are equal if they have equal bin_name, z, dndz, and
        measurement.
        """
        return (
            self.bin_name == other.bin_name
            and np.array_equal(self.z, other.z)
            and np.array_equal(self.dndz, other.dndz)
            and self.measurement == other.measurement
        )


@dataclass(frozen=True, kw_only=True)
class TwoPointXY(YAMLSerializable):
    """Class defining a two-point correlation pair of redshift resolutions.

    It is used to store the two redshift resolutions for the two bins being
    correlated.
    """

    x: InferredGalaxyZDist
    y: InferredGalaxyZDist

    def __post_init__(self) -> None:
        """Make sure the two redshift resolutions are compatible."""
        if not measurement_is_compatible(self.x.measurement, self.y.measurement):
            raise ValueError(
                f"Measurements {self.x.measurement} and {self.y.measurement} "
                f"are not compatible."
            )

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointXY objects."""
        return self.x == other.x and self.y == other.y


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

    def __post_init__(self) -> None:
        """Validate the TwoPointCells data.

        Make sure the ells are a 1D array and X and Y are compatible
        with harmonic-space calculations.
        """
        if len(self.ells.shape) != 1:
            raise ValueError("Ells should be a 1D array.")

        if not measurement_supports_harmonic(
            self.XY.x.measurement
        ) or not measurement_supports_harmonic(self.XY.y.measurement):
            raise ValueError(
                f"Measurements {self.XY.x.measurement} and "
                f"{self.XY.y.measurement} must support harmonic-space calculations."
            )

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointCells objects."""
        return self.XY == other.XY and np.array_equal(self.ells, other.ells)

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x.measurement, self.XY.y.measurement
        )


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

    def __post_init__(self):
        """Validate the TwoPointCWindow data.

        Make sure the window is
        """
        if not isinstance(self.window, Window):
            raise ValueError("Window should be a Window object.")

        if not measurement_supports_harmonic(
            self.XY.x.measurement
        ) or not measurement_supports_harmonic(self.XY.y.measurement):
            raise ValueError(
                f"Measurements {self.XY.x.measurement} and "
                f"{self.XY.y.measurement} must support harmonic-space calculations."
            )

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x.measurement, self.XY.y.measurement
        )


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

    def __post_init__(self):
        """Validate the TwoPointCWindow data.

        Make sure the window is
        """
        if not measurement_supports_real(
            self.XY.x.measurement
        ) or not measurement_supports_real(self.XY.y.measurement):
            raise ValueError(
                f"Measurements {self.XY.x.measurement} and "
                f"{self.XY.y.measurement} must support real-space calculations."
            )

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_real(self.XY.x.measurement, self.XY.y.measurement)

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointXiTheta objects."""
        return self.XY == other.XY and np.array_equal(self.thetas, other.thetas)


# TwoPointXiThetaIndex is a type used to create intermediate objects when
# reading SACC objects. They should not be seen directly by users of Firecrown.
TwoPointXiThetaIndex = TypedDict(
    "TwoPointXiThetaIndex",
    {
        "data_type": str,
        "tracer_names": TracerNames,
        "thetas": npt.NDArray[np.float64],
    },
)


# TwoPointCellsIndex is a type used to create intermediate objects when reading
# SACC objects. They should not be seen directly by users of Firecrown.
TwoPointCellsIndex = TypedDict(
    "TwoPointCellsIndex",
    {
        "data_type": str,
        "tracer_names": TracerNames,
        "ells": npt.NDArray[np.int64],
    },
)


def _extract_candidate_data_types(
    tracer_name: str, data_points: list[sacc.DataPoint]
) -> list[Measurement]:
    """Extract the candidate Measurement for a tracer.

    An exception is raise if the tracer does not have any associated data points.
    """
    tracer_associated_types = {
        d.data_type for d in data_points if tracer_name in d.tracers
    }
    tracer_associated_types_len = len(tracer_associated_types)

    type_count: dict[Measurement, int] = {
        measurement: 0 for measurement in ALL_MEASUREMENTS
    }
    for data_type in tracer_associated_types:
        if data_type not in MEASURED_TYPE_STRING_MAP:
            continue
        a, b = MEASURED_TYPE_STRING_MAP[data_type]

        if a != b:
            type_count[a] += 1
            type_count[b] += 1
        else:
            type_count[a] += 1

    result = [
        measurement
        for measurement, count in type_count.items()
        if count == tracer_associated_types_len
    ]
    if len(result) == 0:
        raise ValueError(
            f"Tracer {tracer_name} does not have data points associated with it. "
            f"Inconsistent SACC object."
        )
    return result


def extract_all_tracers(sacc_data: sacc.Sacc) -> list[InferredGalaxyZDist]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    tracers: list[sacc.tracers.BaseTracer] = sacc_data.tracers.values()

    data_points = sacc_data.get_data_points()

    inferred_galaxy_zdists = []

    for tracer in tracers:
        candidate_measurements = _extract_candidate_data_types(tracer.name, data_points)

        measurement = extract_measurement(candidate_measurements, tracer)

        inferred_galaxy_zdists.append(
            InferredGalaxyZDist(
                bin_name=tracer.name,
                z=tracer.z,
                dndz=tracer.nz,
                measurement=measurement,
            )
        )

    return inferred_galaxy_zdists


def extract_measurement(
    candidate_measurements: list[Galaxies | CMB | Clusters],
    tracer: sacc.tracers.BaseTracer,
) -> Galaxies | CMB | Clusters:
    """Extract from tracer a single type of measurement.

    Only types in candidate_measurements will be considered.
    """
    if len(candidate_measurements) == 1:
        # Only one Measurement appears in all associated data points.
        # We can infer the Measurement from the data points.
        measurement = candidate_measurements[0]
    else:
        # We cannot infer the Measurement from the associated data points.
        # We need to check the tracer name.
        if LENS_REGEX.match(tracer.name):
            if Galaxies.COUNTS not in candidate_measurements:
                raise ValueError(
                    f"Tracer {tracer.name} matches the lens regex but does "
                    f"not have a compatible Measurement. Inconsistent SACC "
                    f"object."
                )
            measurement = Galaxies.COUNTS
        elif SOURCE_REGEX.match(tracer.name):
            # The source tracers can be either shear E or shear T.
            if Galaxies.SHEAR_E in candidate_measurements:
                measurement = Galaxies.SHEAR_E
            elif Galaxies.SHEAR_T in candidate_measurements:
                measurement = Galaxies.SHEAR_T
            else:
                raise ValueError(
                    f"Tracer {tracer.name} matches the source regex but does "
                    f"not have a compatible Measurement. Inconsistent SACC "
                    f"object."
                )
        else:
            raise ValueError(
                f"Tracer {tracer.name} does not have a compatible Measurement. "
                f"Inconsistent SACC object."
            )
    return measurement


def extract_all_data_types_xi_thetas(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
) -> list[TwoPointXiThetaIndex]:
    """Extract all two-point function metadata from a sacc file.

    Extracts the two-point function measurement metadata for all measurements
    made in real space  from a Sacc object.
    """
    tag_name = "theta"

    data_types = sacc_data.get_data_types()

    data_types_xi_thetas = [
        data_type for data_type in data_types if tag_name in required_tags[data_type]
    ]
    if allowed_data_type is not None:
        data_types_xi_thetas = [
            data_type
            for data_type in data_types_xi_thetas
            if data_type in allowed_data_type
        ]

    all_xi_thetas: list[TwoPointXiThetaIndex] = []
    for data_type in data_types_xi_thetas:
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )

            all_xi_thetas.append(
                {
                    "data_type": data_type,
                    "tracer_names": TracerNames(*combo),
                    "thetas": np.array(
                        sacc_data.get_tag(tag_name, data_type=data_type, tracers=combo)
                    ),
                }
            )

    return all_xi_thetas


def extract_all_data_types_cells(
    sacc_data: sacc.Sacc, allowed_data_type: None | list[str] = None
) -> list[TwoPointCellsIndex]:
    """Extracts the two-point function metadata from a sacc file."""
    tag_name = "ell"

    data_types = sacc_data.get_data_types()

    data_types_cells = [
        data_type for data_type in data_types if tag_name in required_tags[data_type]
    ]

    if allowed_data_type is not None:
        data_types_cells = [
            data_type
            for data_type in data_types_cells
            if data_type in allowed_data_type
        ]

    all_cell: list[TwoPointCellsIndex] = []
    for data_type in data_types_cells:
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )

            all_cell.append(
                {
                    "data_type": data_type,
                    "tracer_names": TracerNames(*combo),
                    "ells": np.array(
                        sacc_data.get_tag(tag_name, data_type=data_type, tracers=combo)
                    ).astype(np.int64),
                }
            )

    return all_cell


def extract_all_photoz_bin_combinations(
    sacc_data: sacc.Sacc,
) -> list[TwoPointXY]:
    """Extracts the two-point function metadata from a sacc file."""
    inferred_galaxy_zdists = extract_all_tracers(sacc_data)
    bin_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)

    return bin_combinations


def extract_window_function(
    sacc_data: sacc.Sacc, indices: npt.NDArray[np.int64]
) -> None | Window:
    """Extract a window function from a sacc file that matches the given indices.

    If there is no appropriate window function, return None.
    """
    bandpower_window = sacc_data.get_bandpower_windows(indices)
    if bandpower_window is None:
        return None
    return Window(
        ells=bandpower_window.values,
        weights=bandpower_window.weight / bandpower_window.weight.sum(axis=0),
    )


LENS_REGEX = re.compile(r"^lens\d+$")
SOURCE_REGEX = re.compile(r"^(src\d+|source\d+)$")


def make_all_photoz_bin_combinations(
    inferred_galaxy_zdists: list[InferredGalaxyZDist],
) -> list[TwoPointXY]:
    """Extract the two-point function metadata from a sacc file."""
    bin_combinations = [
        TwoPointXY(x=igz1, y=igz2)
        for igz1, igz2 in combinations_with_replacement(inferred_galaxy_zdists, 2)
        if measurement_is_compatible(igz1.measurement, igz2.measurement)
    ]

    return bin_combinations


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
    if measurement_supports_real(a) and measurement_supports_real(b)
} | {
    type_to_sacc_string_harmonic(a, b): (a, b) if a < b else (b, a)
    for a, b in combinations_with_replacement(ALL_MEASUREMENTS, 2)
    if measurement_supports_harmonic(a) and measurement_supports_harmonic(b)
}
