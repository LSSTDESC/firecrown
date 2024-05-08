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

from firecrown.utils import upper_triangle_indices, compare_optional_arrays

ST = TypeVar("ST")  # This will be used in YAMLSerializable


class YAMLSerializable:
    """Protocol for classes that can be serialized to and from YAML."""

    def to_yaml(self: ST) -> str:
        """Return the YAML representation of the object."""
        return yaml.dump(self, Dumper=Dumper)

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


class GalaxyMeasuredType(YAMLSerializable, str, Enum):
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
        if self == GalaxyMeasuredType.SHEAR_E:
            return "shear"
        if self == GalaxyMeasuredType.SHEAR_T:
            return "shear"
        if self == GalaxyMeasuredType.COUNTS:
            return "density"
        raise ValueError("Untranslated GalaxyMeasuredType encountered")

    def polarization(self) -> str:
        """Return the SACC polarization code.

        This is the third part of the SACC string used to denote the specific
        measurement type.
        """
        if self == GalaxyMeasuredType.SHEAR_E:
            return "e"
        if self == GalaxyMeasuredType.SHEAR_T:
            return "t"
        if self == GalaxyMeasuredType.COUNTS:
            return ""
        raise ValueError("Untranslated GalaxyMeasuredType encountered")

    def __lt__(self, other):
        """Define a comparison function for the GalaxyMeasuredType enumeration."""
        return compare_enums(self, other) < 0

    def __eq__(self, other):
        """Define an equality test for GalaxyMeasuredType enumeration."""
        return compare_enums(self, other) == 0

    def __ne__(self, other):
        """Negation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Define a hash function that uses both type and value information."""
        return hash((GalaxyMeasuredType, self.value))


class CMBMeasuredType(YAMLSerializable, str, Enum):
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
        if self == CMBMeasuredType.CONVERGENCE:
            return "convergence"
        raise ValueError("Untranslated CMBMeasuredType encountered")

    def polarization(self) -> str:
        """Return the SACC polarization code.

        This is the third part of the SACC string used to denote the specific
        measurement type.
        """
        if self == CMBMeasuredType.CONVERGENCE:
            return ""
        raise ValueError("Untranslated CMBMeasuredType encountered")

    def __lt__(self, other):
        """Define a comparison function for the CMBMeasuredType enumeration."""
        return compare_enums(self, other) < 0

    def __eq__(self, other):
        """Define an equality test for CMBMeasuredType enumeration."""
        return compare_enums(self, other) == 0

    def __ne__(self, other):
        """Negation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Define a hash function that uses both type and value information."""
        return hash((CMBMeasuredType, self.value))


class ClusterMeasuredType(YAMLSerializable, str, Enum):
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
        if self == ClusterMeasuredType.COUNTS:
            return "density"
        raise ValueError("Untranslated ClusterMeasuredType encountered")

    def polarization(self) -> str:
        """Return the SACC polarization code.

        This is the third part of the SACC string used to denote the specific
        measurement type.
        """
        if self == ClusterMeasuredType.COUNTS:
            return ""
        raise ValueError("Untranslated ClusterMeasuredType encountered")

    def __lt__(self, other):
        """Define a comparison function for the ClusterMeasuredType enumeration."""
        return compare_enums(self, other) < 0

    def __eq__(self, other):
        """Define an equality test for ClusterMeasuredType enumeration."""
        return compare_enums(self, other) == 0

    def __ne__(self, other):
        """Negation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Define a hash function that uses both type and value information."""
        return hash((ClusterMeasuredType, self.value))


MeasuredType = GalaxyMeasuredType | CMBMeasuredType | ClusterMeasuredType
ALL_MEASURED_TYPES: list[MeasuredType] = list(
    chain(GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType)
)
ALL_MEASURED_TYPES_TYPES = (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType)
HARMONIC_ONLY_MEASURED_TYPES = (GalaxyMeasuredType.SHEAR_E,)
REAL_ONLY_MEASURED_TYPES = (GalaxyMeasuredType.SHEAR_T,)


def measured_type_is_compatible(a: MeasuredType, b: MeasuredType) -> bool:
    """Check if two MeasuredType are compatible.

    Two MeasuredType are compatible if they can be correlated in a two-point function.
    """
    if a in HARMONIC_ONLY_MEASURED_TYPES and b in REAL_ONLY_MEASURED_TYPES:
        return False
    if a in REAL_ONLY_MEASURED_TYPES and b in HARMONIC_ONLY_MEASURED_TYPES:
        return False
    return True


def measured_type_supports_real(x: MeasuredType) -> bool:
    """Return True if x supports real-space calculations."""
    return x not in HARMONIC_ONLY_MEASURED_TYPES


def measured_type_supports_harmonic(x: MeasuredType) -> bool:
    """Return True if x supports harmonic-space calculations."""
    return x not in REAL_ONLY_MEASURED_TYPES


def compare_enums(a: MeasuredType, b: MeasuredType) -> int:
    """Define a comparison function for the MeasuredType enumeration.

    Return -1 if a comes before b, 0 if they are the same, and +1 if b comes before a.
    """
    order = (CMBMeasuredType, ClusterMeasuredType, GalaxyMeasuredType)
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
    measured_type: MeasuredType

    def __post_init__(self) -> None:
        """Validate the redshift resolution data.

        - Make sure the z and dndz arrays have the same shape;
        - The measured type must be a MeasuredType.
        - The bin_name should not be empty.
        """
        if self.z.shape != self.dndz.shape:
            raise ValueError("The z and dndz arrays should have the same shape.")

        if not isinstance(self.measured_type, ALL_MEASURED_TYPES_TYPES):
            raise ValueError("The measured_type should be a MeasuredType.")

        if self.bin_name == "":
            raise ValueError("The bin_name should not be empty.")

    def __eq__(self, other):
        """Equality test for InferredGalaxyZDist.

        Two InferredGalaxyZDist are equal if they have equal bin_name, z, dndz, and
        measured_type.
        """
        return (
            self.bin_name == other.bin_name
            and np.array_equal(self.z, other.z)
            and np.array_equal(self.dndz, other.dndz)
            and self.measured_type == other.measured_type
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
        if not measured_type_is_compatible(self.x.measured_type, self.y.measured_type):
            raise ValueError(
                f"Measured types {self.x.measured_type} and {self.y.measured_type} "
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

        if not measured_type_supports_harmonic(
            self.XY.x.measured_type
        ) or not measured_type_supports_harmonic(self.XY.y.measured_type):
            raise ValueError(
                f"Measured types {self.XY.x.measured_type} and "
                f"{self.XY.y.measured_type} must support harmonic-space calculations."
            )

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointCells objects."""
        return self.XY == other.XY and np.array_equal(self.ells, other.ells)

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x.measured_type, self.XY.y.measured_type
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

        if not measured_type_supports_harmonic(
            self.XY.x.measured_type
        ) or not measured_type_supports_harmonic(self.XY.y.measured_type):
            raise ValueError(
                f"Measured types {self.XY.x.measured_type} and "
                f"{self.XY.y.measured_type} must support harmonic-space calculations."
            )

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x.measured_type, self.XY.y.measured_type
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
        if not measured_type_supports_real(
            self.XY.x.measured_type
        ) or not measured_type_supports_real(self.XY.y.measured_type):
            raise ValueError(
                f"Measured types {self.XY.x.measured_type} and "
                f"{self.XY.y.measured_type} must support real-space calculations."
            )

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_real(
            self.XY.x.measured_type, self.XY.y.measured_type
        )

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
) -> list[MeasuredType]:
    """Extract the candidate measured types for a tracer."""
    tracer_associated_types = {
        d.data_type for d in data_points if tracer_name in d.tracers
    }
    tracer_associated_types_len = len(tracer_associated_types)

    type_count: dict[MeasuredType, int] = {
        measured_type: 0 for measured_type in ALL_MEASURED_TYPES
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

    return [
        measured_type
        for measured_type, count in type_count.items()
        if count == tracer_associated_types_len
    ]


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
        candidate_measured_types = _extract_candidate_data_types(
            tracer.name, data_points
        )

        if len(candidate_measured_types) == 0:
            raise ValueError(
                f"Tracer {tracer.name} does not have data points associated with it. "
                f"Inconsistent SACC object."
            )

        measured_type = extract_measured_type(candidate_measured_types, tracer)

        inferred_galaxy_zdists.append(
            InferredGalaxyZDist(
                bin_name=tracer.name,
                z=tracer.z,
                dndz=tracer.nz,
                measured_type=measured_type,
            )
        )

    return inferred_galaxy_zdists


def extract_measured_type(
    candidate_measured_types: list[
        GalaxyMeasuredType | CMBMeasuredType | ClusterMeasuredType
    ],
    tracer: sacc.tracers.BaseTracer,
) -> GalaxyMeasuredType | CMBMeasuredType | ClusterMeasuredType:
    """Extract from tracer a single type of measurement.

    Only types in candidate_measured_types will be considered.
    """
    if len(candidate_measured_types) == 1:
        # Only one measured type appears in all associated data points.
        # We can infer the measured type from the data points.
        measured_type = candidate_measured_types[0]
    else:
        # We cannot infer the measured type from the associated data points.
        # We need to check the tracer name.
        if LENS_REGEX.match(tracer.name):
            if GalaxyMeasuredType.COUNTS not in candidate_measured_types:
                raise ValueError(
                    f"Tracer {tracer.name} matches the lens regex but does "
                    f"not have a compatible measured type. Inconsistent SACC "
                    f"object."
                )
            measured_type = GalaxyMeasuredType.COUNTS
        elif SOURCE_REGEX.match(tracer.name):
            # The source tracers can be either shear E or shear T.
            if GalaxyMeasuredType.SHEAR_E in candidate_measured_types:
                measured_type = GalaxyMeasuredType.SHEAR_E
            elif GalaxyMeasuredType.SHEAR_T in candidate_measured_types:
                measured_type = GalaxyMeasuredType.SHEAR_T
            else:
                raise ValueError(
                    f"Tracer {tracer.name} matches the source regex but does "
                    f"not have a compatible measured type. Inconsistent SACC "
                    f"object."
                )
        else:
            raise ValueError(
                f"Tracer {tracer.name} does not have a compatible measured type. "
                f"Inconsistent SACC object."
            )
    return measured_type


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
    """Extracts the two-point function metadata from a sacc file."""
    inferred_galaxy_zdists_len = len(inferred_galaxy_zdists)

    bin_combinations = [
        TwoPointXY(
            x=inferred_galaxy_zdists[i],
            y=inferred_galaxy_zdists[j],
        )
        for i, j in upper_triangle_indices(inferred_galaxy_zdists_len)
        if measured_type_is_compatible(
            inferred_galaxy_zdists[i].measured_type,
            inferred_galaxy_zdists[j].measured_type,
        )
    ]

    return bin_combinations


def _type_to_sacc_string_common(x: MeasuredType, y: MeasuredType) -> str:
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


def type_to_sacc_string_real(x: MeasuredType, y: MeasuredType) -> str:
    """Return the final SACC string used to denote the real-space correlation.

    The SACC string used to denote the real-space correlation type
    between measurements of x and y.
    """
    a, b = sorted([x, y])
    suffix = f"{a.polarization()}{b.polarization()}"

    if a in HARMONIC_ONLY_MEASURED_TYPES or b in HARMONIC_ONLY_MEASURED_TYPES:
        raise ValueError("Real-space correlation not supported for shear E.")

    return _type_to_sacc_string_common(x, y) + (f"xi_{suffix}" if suffix else "xi")


def type_to_sacc_string_harmonic(x: MeasuredType, y: MeasuredType) -> str:
    """Return the final SACC string used to denote the harmonic-space correlation.

    the SACC string used to denote the harmonic-space correlation type
    between measurements of x and y.
    """
    a, b = sorted([x, y])
    suffix = f"{a.polarization()}{b.polarization()}"

    if a in REAL_ONLY_MEASURED_TYPES or b in REAL_ONLY_MEASURED_TYPES:
        raise ValueError("Harmonic-space correlation not supported for shear T.")

    return _type_to_sacc_string_common(x, y) + (f"cl_{suffix}" if suffix else "cl")


MEASURED_TYPE_STRING_MAP: dict[str, tuple[MeasuredType, MeasuredType]] = {
    type_to_sacc_string_real(a, b): (a, b) if a < b else (b, a)
    for a, b in combinations_with_replacement(ALL_MEASURED_TYPES, 2)
    if measured_type_supports_real(a) and measured_type_supports_real(b)
} | {
    type_to_sacc_string_harmonic(a, b): (a, b) if a < b else (b, a)
    for a, b in combinations_with_replacement(ALL_MEASURED_TYPES, 2)
    if measured_type_supports_harmonic(a) and measured_type_supports_harmonic(b)
}
