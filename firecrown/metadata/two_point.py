"""This module deals with two-point functions metadata.

It contains all data classes and functions for store and extract two-point functions
metadata from a sacc file.
"""

from itertools import combinations_with_replacement
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
    ALL_MEASUREMENT_TYPES,
    HARMONIC_ONLY_MEASUREMENTS,
    REAL_ONLY_MEASUREMENTS,
    ALL_MEASUREMENTS,
)

LENS_REGEX = re.compile(r"^lens\d+$")
SOURCE_REGEX = re.compile(r"^(src\d+|source\d+)$")


def make_measurement_dict(value: Measurement) -> dict[str, str]:
    """Create a dictionary from a Measurement object.

    :param value: the measurement to turn into a dictionary
    """
    return {"subject": type(value).__name__, "property": value.name}


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

    def __str__(self) -> str:
        """Return a string representation of the TwoPointXY object."""
        return f"({self.x.bin_name}, {self.y.bin_name})"

    def get_tracer_names(self) -> TracerNames:
        """Return the TracerNames object for the TwoPointXY object."""
        return TracerNames(self.x.bin_name, self.y.bin_name)


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
            self.XY.x.measurement
        ) or not measurement_supports_harmonic(self.XY.y.measurement):
            raise ValueError(
                f"Measurements {self.XY.x.measurement} and "
                f"{self.XY.y.measurement} must support harmonic-space calculations."
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
            self.XY.x.measurement, self.XY.y.measurement
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

        XY = TwoPointXY(
            x=inferred_galaxy_zdists_dict[tracer_names[0]],
            y=inferred_galaxy_zdists_dict[tracer_names[1]],
        )

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

        XY = TwoPointXY(
            x=inferred_galaxy_zdists_dict[tracer_names[0]],
            y=inferred_galaxy_zdists_dict[tracer_names[1]],
        )

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
