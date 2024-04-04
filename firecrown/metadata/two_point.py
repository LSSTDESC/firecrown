"""This module deals with two-point functions metadata.

It contains all data classes and functions for store and extract two-point functions
metadata from a sacc file.
"""

from typing import TypedDict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import re
from itertools import chain

import numpy as np
import numpy.typing as npt

import sacc
from sacc.data_types import required_tags

from firecrown.utils import upper_triangle_indices


@dataclass(frozen=True)
class TracerNames:
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


class GalaxyMeasuredType(str, Enum):
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


class CMBMeasuredType(str, Enum):
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


class ClusterMeasuredType(str, Enum):
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


MeasuredType = Union[GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType]
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


ALL_MEASURED_TYPES: list[MeasuredType] = list(
    chain(GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType)
)


# kw_only=True only available in Python >= 3.10:
# TODO update when we drop Python 3.9
@dataclass(frozen=True)
class InferredGalaxyZDist:
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


# kw_only=True only available in Python >= 3.10:
# TODO update when we drop Python 3.9
@dataclass(frozen=True)
class TwoPointXY:
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


# kw_only=True only available in Python >= 3.10:
# TODO update when we drop Python 3.9
@dataclass(frozen=True)
class TwoPointCells:
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

        if not measured_type_supports_harmonic(self.XY.x.measured_type):
            raise ValueError(
                f"Measured type {self.XY.x.measured_type} does not "
                f"support harmonic-space calculations."
            )

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x.measured_type, self.XY.y.measured_type
        )


# kw_only=True only available in Python >= 3.10:
# TODO update when we drop Python 3.9
@dataclass()
class Window:
    """The class used to represent a window function.

    It contains the ells at which the window function is defined, the weights
    of the window function, and the ells at which the window function is
    interpolated.

    It may contain the ells for interpolation if the theory prediction is
    calculated at a different set of ells than the window function.
    """

    ells: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]
    ells_for_interpolation: Optional[npt.NDArray[np.int64]] = None

    def __post_init__(self) -> None:
        """Make sure the weights have the right shape."""
        if len(self.weights.shape) != 2:
            raise ValueError("Weights should be a 2D array.")
        if self.weights.shape[0] != len(self.ells):
            raise ValueError("Weights should have the same number of rows as ells.")

    def n_observations(self) -> int:
        """Return the number of observations supported by the window function."""
        return self.weights.shape[1]


# kw_only=True only available in Python >= 3.10:
# TODO update when we drop Python 3.9
@dataclass(frozen=True)
class TwoPointCWindow:
    """Two-point function with a window function.

    The class used to store the metadata for a (spherical) harmonic-space two-point
    function measured on a sphere, with an associated window function.

    This includes the two redshift resolutions (one for each binned quantity) and the
    matrix (window function) that relates the measured Cl's with the predicted Cl's.

    Note that the matrix `window` always has l=0 and l=1 suppressed.
    """

    XY: TwoPointXY
    window: npt.NDArray[np.int64]

    def __post_init__(self):
        """Validate the TwoPointCWindow data.

        Make sure the window has the right shape. Check if the type of XY is compatible
        with harmonic-space calculations.
        """
        if len(self.window.shape) != 2:
            raise ValueError("Window should be a 2D array.")

        if not measured_type_supports_harmonic(self.XY.x.measured_type):
            raise ValueError(
                f"Measured type {self.XY.x.measured_type} does not "
                f"support harmonic-space calculations."
            )


# kw_only=True only available in Python >= 3.10:
# TODO update when we drop Python 3.9
@dataclass(frozen=True)
class TwoPointXiTheta:
    """Class defining the metadata for a real-space two-point measurement.

    The class used to store the metadata for a real-space two-point function measured
    on a sphere.

    This includes the two redshift resolutions (one for each binned quantity) and the a
    array of (floating point) theta (angle) values at which the two-point function
    which has  this metadata were calculated.
    """

    XY: TwoPointXY
    theta: npt.NDArray[np.float64]


# TwoPointXiThetaIndex is a type used to create intermediate objects when
# reading SACC objects. They should not be seen directly by users of Firecrown.
TwoPointXiThetaIndex = TypedDict(
    "TwoPointXiThetaIndex",
    {
        "data_type": str,
        "tracer_names": TracerNames,
        "theta": npt.NDArray[np.float64],
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


def extract_all_tracers(sacc_data: sacc.Sacc) -> list[sacc.tracers.BaseTracer]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    return sacc_data.tracers.values()


def extract_all_data_types_xi_thetas(
    sacc_data: sacc.Sacc,
    allowed_data_type: Optional[list[Tuple[MeasuredType, MeasuredType]]] = None,
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
                    "theta": sacc_data.get_tag(
                        tag_name, data_type=data_type, tracers=combo
                    ),
                }
            )

    return all_xi_thetas


def extract_all_data_types_cells(
    sacc_data: sacc.Sacc, allowed_data_type: Optional[list[str]] = None
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
                    "ells": sacc_data.get_tag(
                        tag_name, data_type=data_type, tracers=combo
                    ),
                }
            )

    return all_cell


def extract_all_photoz_bin_combinations(
    sacc_data: sacc.Sacc,
) -> list[TwoPointXY]:
    """Extracts the two-point function metadata from a sacc file."""
    tracers = extract_all_tracers(sacc_data)
    inferred_galaxy_zdists = make_galaxy_zdists_dataclasses(tracers)
    bin_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)

    return bin_combinations


def extract_window_function(
    sacc_data: sacc.Sacc, indices: npt.NDArray[np.int64]
) -> Optional[Window]:
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


def measured_type_from_name(name: str) -> MeasuredType:
    """Extracts the measured type from the tracer name."""
    if LENS_REGEX.match(name):
        return GalaxyMeasuredType.COUNTS

    if SOURCE_REGEX.match(name):
        return GalaxyMeasuredType.SHEAR_E

    raise ValueError(f"Measured type not found for tracer name {name}.")


def make_galaxy_zdists_dataclasses(
    tracers: list[sacc.tracers.BaseTracer],
) -> list[InferredGalaxyZDist]:
    """Make a list of InferredGalaxyZDist dataclasses from a list of sacc tracers."""
    inferrerd_galaxy_zdists = [
        InferredGalaxyZDist(
            bin_name=tracer.name,
            z=tracer.z,
            dndz=tracer.nz,
            measured_type=measured_type_from_name(tracer.name),
        )
        for tracer in tracers
        if isinstance(tracer, sacc.tracers.NZTracer)
    ]

    return inferrerd_galaxy_zdists


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
    ]

    return bin_combinations


DATATYPE_DICT = {
    (
        GalaxyMeasuredType.COUNTS,
        GalaxyMeasuredType.COUNTS,
        "xi_theta",
    ): "galaxy_density_xi",
    (
        GalaxyMeasuredType.SHEAR_E,
        GalaxyMeasuredType.SHEAR_E,
        "xi",
    ): "galaxy_shear_xi_plus",
}


def make_xi_thetas(
    *,
    data_type: str,
    tracer_names: TracerNames,
    theta: np.ndarray,
    bin_combinations: list[TwoPointXY],
) -> TwoPointXiTheta:
    """Make a TwoPointXiTheta dataclass from the two-point function metadata."""
    bin_combo = get_combination(bin_combinations, tracer_names)

    assert (
        DATATYPE_DICT[
            (bin_combo.x.measured_type, bin_combo.y.measured_type, "xi_theta")
        ]
        == data_type
    )

    return TwoPointXiTheta(
        XY=bin_combo,
        theta=theta,
    )


def make_cells(
    *,
    data_type: str,
    tracer_names: TracerNames,
    ells: np.ndarray,
    bin_combinations: list[TwoPointXY],
) -> TwoPointCells:
    """Make a TwoPointCells dataclass from the two-point function metadata."""
    bin_combo = get_combination(bin_combinations, tracer_names)

    assert (
        DATATYPE_DICT[
            (bin_combo.x.measured_type, bin_combo.y.measured_type, "xi_theta")
        ]
        == data_type
    )

    return TwoPointCells(
        XY=bin_combo,
        ells=ells,
    )


def get_combination(bin_combinations: list[TwoPointXY], bin_combo: TracerNames):
    """Get the combination of two-point function data for a sacc file."""
    for combination in bin_combinations:
        if (
            combination.x.bin_name == bin_combo[0]
            and combination.y.bin_name == bin_combo[1]
        ):
            return combination

        if (
            combination.x.bin_name == bin_combo[1]
            and combination.y.bin_name == bin_combo[0]
        ):
            return combination

    raise ValueError(f"Combination {bin_combo} not found in bin_combinations.")


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
