"""This module contains all data classes and functions for store and extract two-point
functions metadata from a sacc file."""

from typing import TypedDict, Optional, Union
from dataclasses import dataclass
from enum import StrEnum, auto
import re

import numpy as np
import numpy.typing as npt

import sacc
from sacc.data_types import required_tags

from firecrown.utils import upper_triangle_indices
from firecrown.likelihood.gauss_family.statistic.two_point import TracerNames


class GalaxyMeasuredType(StrEnum):
    """This enumeration type provides identifiers for the different types of
    galaxy-related types of measurement.

    SACC has some notion of supporting other types, but incomplete
    implementation. When support for more types is added to SACC this
    enumeration needs to be updated.
    """

    CONVERGENCE = auto()
    COUNTS = auto()
    SHEAR_E = auto()


class CMBMeasuredType(StrEnum):
    """This enumeration type provides identifiers for the different types of
    CMB-related types of measurement.

    SACC has some notion of supporting other types, but incomplete
    implementation. When support for more types is added to SACC this
    enumeration needs to be updated.
    """

    CONVERGENCE = auto()
    TEMPERATURE = auto()
    POLARIATION_E = auto()
    POLARIATION_B = auto()


MeasuredType = Union[GalaxyMeasuredType, CMBMeasuredType]


@dataclass(frozen=True, kw_only=True)
class InferredGalaxyZDist:
    """The class used to store the redshift resolution data for a sacc file.

    The sacc file is a complicated set of tracers (bins) and surveys. This class
    is used to store the redshift resolution data for a single photometric bin.
    """

    bin_name: str
    z: np.ndarray
    dndz: np.ndarray
    measured_type: MeasuredType


@dataclass(frozen=True, kw_only=True)
class TwoPointXY:
    """The class used to store the two redshift resolutions for the two bins
    being correlated."""

    x: InferredGalaxyZDist
    y: InferredGalaxyZDist


@dataclass(frozen=True, kw_only=True)
class TwoPointCells:
    """The class used to store the metadata for a (spherical) harmonic-space
    two-point function measured on a sphere.

    This includes the two redshift resolutions (one for each binned quantity)
    and the array of (integer) l's at which the two-point function which has
    this metadata were calculated.
    """

    XY: TwoPointXY
    ells: npt.NDArray[np.int64]


@dataclass(frozen=True, kw_only=True)
class TwoPointCWindow:
    """The class used to store the metadata for a (spherical) harmonic-space
    two-point function measured on a sphere, with an associated window
    function.

    This includes the two redshift resolutions (one for each binned quantity)
    and the matrix (window function) that relates the measured Cl's with the
    predicted Cl's.

    Note that the matrix `window` always has l=0 and l=1 suppressed.
    """

    XY: TwoPointXY
    window: npt.NDArray[np.int64]

    def __post_init__(self):
        if len(self.window.shape) != 2:
            raise ValueError("Window should be a 2D array.")


@dataclass(frozen=True, kw_only=True)
class TwoPointXiTheta:
    """The class used to store the metadata for a real-space two-point
    function measured on a sphere.

    This includes the two redshift resolutions (one for each binned quantity)
    and the a array of (floating point) theta (angle) values at which the
    two-point function which has  this metadata were calculated.
    """

    XY: TwoPointXY
    theta: npt.NDArray[np.float64]


TwoPointXiThetaIndex = TypedDict(
    "TwoPointXiThetaIndex",
    {
        "tracer_names": TracerNames,
        "theta": npt.NDArray[np.float64],
    },
)


TwoPointCellsIndex = TypedDict(
    "TwoPointCellsIndex",
    {
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
    allowed_data_type: Optional[list[str]] = None,
) -> list[TwoPointXiThetaIndex]:
    """Extracts the two-point function measurement metadata for all measurements
    made in real space  from a Sacc object."""

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
    tracer_names: TracerNames,
    ells: np.ndarray,
    bin_combinations: list[TwoPointXY],
) -> TwoPointCells:
    """Make a TwoPointCells dataclass from the two-point function metadata."""

    bin_combo = get_combination(bin_combinations, tracer_names)

    return TwoPointCells(
        XY=bin_combo,
        ells=ells,
    )


def get_combination(bin_combinations: list[TwoPointXY], bin_combo: TracerNames):
    """Get the combination of two-point function data for a sacc file."""

    for combination in bin_combinations:
        if combination.x.bin_name == bin_combo[0] and combination.y.bin_name == bin_combo[1]:
            return combination

        if combination.x.bin_name == bin_combo[1] and combination.y.bin_name == bin_combo[0]:
            return combination

    raise ValueError(f"Combination {bin_combo} not found in bin_combinations.")
