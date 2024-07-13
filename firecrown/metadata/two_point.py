"""This module deals with two-point functions metadata.

It contains all data classes and functions for store and extract two-point functions
metadata from a sacc file.
"""

from itertools import combinations_with_replacement, product
import hashlib
from typing import TypedDict, Sequence
from dataclasses import dataclass
import re

import numpy as np
import numpy.typing as npt

import sacc
from sacc.data_types import required_tags

from firecrown.utils import compare_optional_arrays, compare_optionals, YAMLSerializable
from firecrown.metadata.two_point_types import (
    Galaxies,
    TracerNames,
    CMB,
    Clusters,
    Measurement,
    InferredGalaxyZDist,
    TwoPointMeasurement,
    HARMONIC_ONLY_MEASUREMENTS,
    REAL_ONLY_MEASUREMENTS,
    EXACT_MATCH_MEASUREMENTS,
    ALL_MEASUREMENTS,
)

LENS_REGEX = re.compile(r"^lens\d+$")
SOURCE_REGEX = re.compile(r"^(src\d+|source\d+)$")


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
            if len(self.Cell.data.shape) != 1:
                raise ValueError("Data should be a 1D array.")
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

GALAXY_SOURCE_TYPES = (
    Galaxies.SHEAR_E,
    Galaxies.SHEAR_T,
    Galaxies.SHEAR_MINUS,
    Galaxies.SHEAR_PLUS,
)
GALAXY_LENS_TYPES = (Galaxies.COUNTS,)


def _extract_all_candidate_data_types(
    data_points: list[sacc.DataPoint],
    include_maybe_types: bool = False,
) -> dict[str, set[Measurement]]:
    """Extract all candidate Measurement from the data points.

    The candidate Measurement are the ones that appear in the data points.
    """
    all_data_types: set[tuple[str, str, str]] = {
        (d.data_type, d.tracers[0], d.tracers[1]) for d in data_points
    }
    sure_types, maybe_types = _extract_sure_and_maybe_types(all_data_types)

    # Remove the sure types from the maybe types.
    for tracer0, sure_types0 in sure_types.items():
        maybe_types[tracer0] -= sure_types0

    # Filter maybe types.
    for data_type, tracer1, tracer2 in all_data_types:
        if data_type not in MEASURED_TYPE_STRING_MAP:
            continue
        a, b = MEASURED_TYPE_STRING_MAP[data_type]

        if a == b:
            continue

        if a in sure_types[tracer1] and b in sure_types[tracer2]:
            maybe_types[tracer1].discard(b)
            maybe_types[tracer2].discard(a)
        elif a in sure_types[tracer2] and b in sure_types[tracer1]:
            maybe_types[tracer1].discard(a)
            maybe_types[tracer2].discard(b)

    if include_maybe_types:
        return {
            tracer0: sure_types0 | maybe_types[tracer0]
            for tracer0, sure_types0 in sure_types.items()
        }
    return sure_types


def _extract_sure_and_maybe_types(all_data_types):
    sure_types: dict[str, set[Measurement]] = {}
    maybe_types: dict[str, set[Measurement]] = {}

    for data_type, tracer1, tracer2 in all_data_types:
        sure_types[tracer1] = set()
        sure_types[tracer2] = set()
        maybe_types[tracer1] = set()
        maybe_types[tracer2] = set()

    # Getting the sure and maybe types for each tracer.
    for data_type, tracer1, tracer2 in all_data_types:
        if data_type not in MEASURED_TYPE_STRING_MAP:
            continue
        a, b = MEASURED_TYPE_STRING_MAP[data_type]

        if a == b:
            sure_types[tracer1].update({a})
            sure_types[tracer2].update({a})
        else:
            name_match, n1, a, n2, b = match_name_type(tracer1, tracer2, a, b)
            if name_match:
                sure_types[n1].update({a})
                sure_types[n2].update({b})
            if not name_match:
                maybe_types[tracer1].update({a, b})
                maybe_types[tracer2].update({a, b})
    return sure_types, maybe_types


def match_name_type(
    tracer1: str,
    tracer2: str,
    a: Measurement,
    b: Measurement,
    require_convetion: bool = False,
) -> tuple[bool, str, Measurement, str, Measurement]:
    for n1, n2 in ((tracer1, tracer2), (tracer2, tracer1)):
        if LENS_REGEX.match(n1) and SOURCE_REGEX.match(n2):
            if a in GALAXY_SOURCE_TYPES and b in GALAXY_LENS_TYPES:
                return True, n1, b, n2, a
            if b in GALAXY_SOURCE_TYPES and a in GALAXY_LENS_TYPES:
                return True, n1, a, n2, b
            raise ValueError(
                "Invalid SACC file, tracer names do not respect "
                "the naming convetion."
            )
    if require_convetion:
        if LENS_REGEX.match(tracer1) and LENS_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b
        if SOURCE_REGEX.match(tracer1) and SOURCE_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b

        raise ValueError(
            f"Invalid SACC file, tracer names ({tracer1}, {tracer2}) "
            f"do not respect the naming convetion."
        )

    return False, tracer1, a, tracer2, b


def extract_all_tracers(sacc_data: sacc.Sacc) -> list[InferredGalaxyZDist]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    tracers: list[sacc.tracers.BaseTracer] = sacc_data.tracers.values()
    tracer_types = extract_all_tracers_types(sacc_data)
    for tracer0, tracer_types0 in tracer_types.items():
        if len(tracer_types0) == 0:
            raise ValueError(
                f"Tracer {tracer0} does not have data points associated with it. "
                f"Inconsistent SACC object."
            )

    return [
        InferredGalaxyZDist(
            bin_name=tracer.name,
            z=tracer.z,
            dndz=tracer.nz,
            measurements=tracer_types[tracer.name],
        )
        for tracer in tracers
    ]


def extract_all_tracers_types(
    sacc_data: sacc.Sacc,
    include_maybe_types: bool = False,
) -> dict[str, set[Measurement]]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    data_points = sacc_data.get_data_points()

    return _extract_all_candidate_data_types(data_points, include_maybe_types)


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


def _build_two_point_xy(
    inferred_galaxy_zdists_dict, tracer_names, data_type
) -> TwoPointXY:
    """Build a TwoPointXY object from the inferred galaxy z distributions.

    The TwoPointXY object is built from the inferred galaxy z distributions, the data
    type, and the tracer names.
    """
    a, b = MEASURED_TYPE_STRING_MAP[data_type]

    igz1 = inferred_galaxy_zdists_dict[tracer_names[0]]
    igz2 = inferred_galaxy_zdists_dict[tracer_names[1]]

    ab = a in igz1.measurements and b in igz2.measurements
    ba = b in igz1.measurements and a in igz2.measurements
    if a != b and ab and ba:
        raise ValueError(
            f"Ambiguous measurements for tracers {tracer_names}."
            f"Impossible to determine which measurement is from which tracer."
        )
    XY = TwoPointXY(
        x=igz1, y=igz2, x_measurement=a if ab else b, y_measurement=b if ab else a
    )

    return XY


def extract_all_data_cells(
    sacc_data: sacc.Sacc, allowed_data_type: None | list[str] = None
) -> tuple[list[TwoPointCells], list[TwoPointCWindow]]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz for igz in extract_all_tracers(sacc_data)
    }

    cov_hash = hashlib.sha256(sacc_data.covariance.dense).hexdigest()

    two_point_cells = []
    two_point_cwindows = []
    for cell_index in extract_all_data_types_cells(sacc_data, allowed_data_type):
        tracer_names = cell_index["tracer_names"]
        ells = cell_index["ells"]
        data_type = cell_index["data_type"]

        XY = _build_two_point_xy(inferred_galaxy_zdists_dict, tracer_names, data_type)

        ells, Cells, indices = sacc_data.get_ell_cl(
            data_type=data_type,
            tracer1=tracer_names[0],
            tracer2=tracer_names[1],
            return_cov=False,
            return_ind=True,
        )

        Cell = TwoPointMeasurement(
            data=Cells,
            indices=indices,
            covariance_name=cov_hash,
        )

        window = extract_window_function(sacc_data, indices)
        if window is not None:
            two_point_cwindows.append(TwoPointCWindow(XY=XY, window=window, Cell=Cell))
        else:
            two_point_cells.append(TwoPointCells(XY=XY, ells=ells, Cell=Cell))

    return two_point_cells, two_point_cwindows


def extract_all_data_xi_thetas(
    sacc_data: sacc.Sacc, allowed_data_type: None | list[str] = None
) -> list[TwoPointXiTheta]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz for igz in extract_all_tracers(sacc_data)
    }

    cov_hash = hashlib.sha256(sacc_data.covariance.dense).hexdigest()

    two_point_xi_thetas = []
    for xi_theta_index in extract_all_data_types_xi_thetas(
        sacc_data, allowed_data_type
    ):
        tracer_names = xi_theta_index["tracer_names"]
        thetas = xi_theta_index["thetas"]
        data_type = xi_theta_index["data_type"]

        XY = _build_two_point_xy(inferred_galaxy_zdists_dict, tracer_names, data_type)

        thetas, Xis, indices = sacc_data.get_theta_xi(
            data_type=data_type,
            tracer1=tracer_names[0],
            tracer2=tracer_names[1],
            return_cov=False,
            return_ind=True,
        )

        Xi = TwoPointMeasurement(
            data=Xis,
            indices=indices,
            covariance_name=cov_hash,
        )

        two_point_xi_thetas.append(TwoPointXiTheta(XY=XY, thetas=thetas, xis=Xi))

    return two_point_xi_thetas


def check_two_point_consistence_harmonic(
    two_point_cells: Sequence[TwoPointCells | TwoPointCWindow],
) -> None:
    """Check the indices of the harmonic-space two-point functions.

    Make sure the indices of the harmonic-space two-point functions are consistent.
    """
    all_indices_set: set[int] = set()
    index_set_list = []
    cov_name: None | str = None

    for two_point_cell in two_point_cells:
        if two_point_cell.Cell is None:
            raise ValueError(
                f"The TwoPointCells {two_point_cell} does not contain a data."
            )
        if cov_name is None:
            cov_name = two_point_cell.Cell.covariance_name
        elif cov_name != two_point_cell.Cell.covariance_name:
            raise ValueError(
                f"The TwoPointCells {two_point_cell} has a different covariance name "
                f"{two_point_cell.Cell.covariance_name} than the previous "
                f"TwoPointCells {cov_name}."
            )
        index_set = set(two_point_cell.Cell.indices)
        index_set_list.append(index_set)
        if len(index_set) != len(two_point_cell.Cell.indices):
            raise ValueError(
                f"The indices of the TwoPointCells {two_point_cell} are not unique."
            )

        if all_indices_set & index_set:
            for i, index_set_a in enumerate(index_set_list):
                if index_set_a & index_set:
                    raise ValueError(
                        f"The indices of the TwoPointCells {two_point_cells[i]} and "
                        f"{two_point_cell} are not unique."
                    )
        all_indices_set.update(index_set)


def check_two_point_consistence_real(
    two_point_xi_thetas: Sequence[TwoPointXiTheta],
) -> None:
    """Check the indices of the real-space two-point functions.

    Make sure the indices of the real-space two-point functions are consistent.
    """
    all_indices_set: set[int] = set()
    index_set_list = []
    cov_name: None | str = None

    for two_point_xi_theta in two_point_xi_thetas:
        if two_point_xi_theta.xis is None:
            raise ValueError(
                f"The TwoPointXiTheta {two_point_xi_theta} does not contain a data."
            )
        if cov_name is None:
            cov_name = two_point_xi_theta.xis.covariance_name
        elif cov_name != two_point_xi_theta.xis.covariance_name:
            raise ValueError(
                f"The TwoPointXiTheta {two_point_xi_theta} has a different covariance "
                f"name {two_point_xi_theta.xis.covariance_name} than the previous "
                f"TwoPointXiTheta {cov_name}."
            )
        index_set = set(two_point_xi_theta.xis.indices)
        index_set_list.append(index_set)
        if len(index_set) != len(two_point_xi_theta.xis.indices):
            raise ValueError(
                f"The indices of the TwoPointXiTheta {two_point_xi_theta} "
                f"are not unique."
            )

        if all_indices_set & index_set:
            for i, index_set_a in enumerate(index_set_list):
                if index_set_a & index_set:
                    raise ValueError(
                        f"The indices of the TwoPointXiTheta {two_point_xi_thetas[i]} "
                        f"and {two_point_xi_theta} are not unique."
                    )
        all_indices_set.update(index_set)


def make_all_photoz_bin_combinations(
    inferred_galaxy_zdists: list[InferredGalaxyZDist],
) -> list[TwoPointXY]:
    """Extract the two-point function metadata from a sacc file."""
    bin_combinations = [
        TwoPointXY(
            x=igz1, y=igz2, x_measurement=x_measurement, y_measurement=y_measurement
        )
        for igz1, igz2 in combinations_with_replacement(inferred_galaxy_zdists, 2)
        for x_measurement, y_measurement in product(
            igz1.measurements, igz2.measurements
        )
        if measurement_is_compatible(x_measurement, y_measurement)
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


def measurements_from_index(
    index: TwoPointXiThetaIndex | TwoPointCellsIndex,
) -> tuple[str, Measurement, str, Measurement]:
    """Return the measurements from a TwoPointXiThetaIndex object."""
    a, b = MEASURED_TYPE_STRING_MAP[index["data_type"]]
    _, n1, a, n2, b = match_name_type(
        index["tracer_names"].name1,
        index["tracer_names"].name2,
        a,
        b,
        require_convetion=True,
    )
    return n1, a, n2, b
