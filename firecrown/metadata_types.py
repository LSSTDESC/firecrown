"""This module deals with metadata types.

This module contains metadata types definitions.
"""

from abc import abstractmethod
from typing import Any, Annotated
from itertools import chain, combinations_with_replacement
from dataclasses import dataclass
import re
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from itertools import chain, combinations_with_replacement
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    GetCoreSchemaHandler,
    SerializeAsAny,
    ValidatorFunctionWrapHandler,
)
from pydantic_core import core_schema, PydanticUndefined
import numpy as np
import numpy.typing as npt

from firecrown.utils import YAMLSerializable, compare_optional_arrays


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


class TypeSource(str):
    """String to specify the subtype or origin of a measurement source.

    This helps distinguish between different categories of sources within the same
    measurement type. For example:

    - In galaxy counts, this could differentiate between red and blue galaxies.
    - In CMB lensing, it could identify data from different instruments like Planck or
      SPT.
    """

    DEFAULT: "TypeSource"

    def __new__(cls, value):
        """Create a new TypeSource."""
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the TypeSource class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


TypeSource.DEFAULT = TypeSource("default")


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

        :return: True if the measurement is a shear measurement, False otherwise
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
REAL_ONLY_MEASUREMENTS = (
    Galaxies.SHEAR_T,
    Galaxies.PART_OF_XI_MINUS,
    Galaxies.PART_OF_XI_PLUS,
)
EXACT_MATCH_MEASUREMENTS = (Galaxies.PART_OF_XI_MINUS, Galaxies.PART_OF_XI_PLUS)
INCOMPATIBLE_MEASUREMENTS = (Galaxies.SHEAR_T,)
LENS_REGEX = re.compile(r"^lens\d+$")
SOURCE_REGEX = re.compile(r"^(src\d+|source\d+)$")
GALAXY_SOURCE_TYPES = (
    Galaxies.SHEAR_E,
    Galaxies.SHEAR_T,
    Galaxies.PART_OF_XI_MINUS,
    Galaxies.PART_OF_XI_PLUS,
)
GALAXY_LENS_TYPES = (Galaxies.COUNTS,)
CMB_TYPES = (CMB.CONVERGENCE,)


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
    type_source: TypeSource = TypeSource.DEFAULT

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
    if a in INCOMPATIBLE_MEASUREMENTS and b in INCOMPATIBLE_MEASUREMENTS:
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
class TwoPointHarmonic(YAMLSerializable):
    """Class defining the metadata for an harmonic-space two-point measurement.

    The class used to store the metadata for a (spherical) harmonic-space two-point
    function measured on a sphere.

    This includes the two redshift resolutions (one for each binned quantity) and the
    array of (integer) l's at which the two-point function which has this metadata were
    calculated.
    """

    XY: TwoPointXY
    ells: npt.NDArray[np.int64]
    window: None | npt.NDArray[np.float64] = None
    window_ells: None | npt.NDArray[np.float64] = None

    def __post_init__(self) -> None:
        """Validate the TwoPointHarmonic data.

        Make sure the ells are a 1D array and X and Y are compatible
        with harmonic-space calculations.
        """
        if len(self.ells.shape) != 1:
            raise ValueError("Ells should be a 1D array.")

        self._check_window_consistency()

        if not measurement_supports_harmonic(
            self.XY.x_measurement
        ) or not measurement_supports_harmonic(self.XY.y_measurement):
            raise ValueError(
                f"Measurements {self.XY.x_measurement} and "
                f"{self.XY.y_measurement} must support harmonic-space calculations."
            )

    def _check_window_consistency(self) -> None:
        """Make sure the window is consistent with the ells."""
        if self.window is not None:
            if not isinstance(self.window, np.ndarray):
                raise ValueError("window should be a ndarray.")
            if len(self.window.shape) != 2:
                raise ValueError("window should be a 2D array.")
            if self.window.shape[0] != len(self.ells):
                raise ValueError("window should have the same number of rows as ells.")
            if self.window_ells is None:
                raise ValueError("window_ells must be set if window is set.")
            if len(self.window_ells.shape) != 1:
                raise ValueError("window_ells should be a 1D array.")
            if self.window_ells.shape[0] != self.window.shape[1]:
                raise ValueError(
                    "window_ells should have the same number of "
                    "elements as the columns of window."
                )
        elif self.window_ells is not None:
            raise ValueError("window_ells must be None if window is None.")

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointHarmonic objects."""
        if not isinstance(other, TwoPointHarmonic):
            raise ValueError("Can only compare TwoPointHarmonic objects.")

        return (
            self.XY == other.XY
            and np.array_equal(self.ells, other.ells)
            and compare_optional_arrays(self.window, other.window)
        )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointHarmonic object."""
        return f"{self.XY}[{self.get_sacc_name()}]"

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_harmonic(
            self.XY.x_measurement, self.XY.y_measurement
        )

    def n_observations(self) -> int:
        """Return the number of observations described by these metadata.

        :return: The number of observations.
        """
        if self.window is None:
            return self.ells.shape[0]
        return self.window.shape[1]


@dataclass(frozen=True, kw_only=True)
class TwoPointReal(YAMLSerializable):
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
        """Validate the TwoPointReal data.

        Make sure the window is
        """
        if len(self.thetas.shape) != 1:
            raise ValueError("Thetas should be a 1D array.")

        if not measurement_supports_real(
            self.XY.x_measurement
        ) or not measurement_supports_real(self.XY.y_measurement):
            raise ValueError(
                f"Measurements {self.XY.x_measurement} and "
                f"{self.XY.y_measurement} must support real-space calculations."
            )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointReal object."""
        return f"{self.XY}[{self.get_sacc_name()}]"

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return type_to_sacc_string_real(self.XY.x_measurement, self.XY.y_measurement)

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointReal objects."""
        if not isinstance(other, TwoPointReal):
            raise ValueError("Can only compare TwoPointReal objects.")

        return self.XY == other.XY and np.array_equal(self.thetas, other.thetas)

    def n_observations(self) -> int:
        """Return the number of observations described by these metadata.

        :return: The number of observations.
        """
        return self.thetas.shape[0]


class TwoPointCorrelationSpace(YAMLSerializable, StrEnum):
    """This class defines the two-point correlation space.

    The two-point correlation space can be either real or harmonic. The real space
    corresponds measurements in terms of angular separation, while the harmonic space
    corresponds to measurements in terms of spherical harmonics decomposition.
    """

    REAL = auto()
    HARMONIC = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the TypeSource class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


class TwoPointFilterMethod(YAMLSerializable, StrEnum):
    """Defines methods for filtering two-point measurements.

    When filtering a two-point measurement with an associated window, the filter is
    applied to the window first. The user must then choose how to proceed:

    - If filtering by `LABEL`, the `window_ells` labels are used to determine whether
      the observation should be kept.
    - If filtering by `SUPPORT`, the full ell support must lie within the allowed range.
    - If filtering by `SUPPORT_95`, only the 95% support region must lie within the
      range.

    In all cases, filters define an interval of ells to retain.
    """

    LABEL = auto()
    SUPPORT = auto()
    SUPPORT_95 = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the TypeSource class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


RULE_REGISTRY: dict[str, type["BinRule"]] = {}


def register_rule(cls: type["BinRule"]) -> type["BinRule"]:
    """Register a new bin rule class using its Pydantic `kind` default."""
    assert issubclass(cls, BaseModel)
    kind_field = cls.model_fields.get("kind")
    assert kind_field is not None
    if kind_field.default is PydanticUndefined:
        raise ValueError(f"{cls.__name__} has no default for 'kind'")
    kind_value = kind_field.default
    RULE_REGISTRY[kind_value] = cls
    return cls


ZDistPair = tuple[InferredGalaxyZDist, InferredGalaxyZDist]
MeasurementPair = tuple[Measurement, Measurement]


class BinRule(BaseModel):
    """Class defining the bin combinator for two-point measurements.

    The bin combinator is used to combine several `InferredGalaxyZDist` into
    `TwoPointXY` objects.
    """

    kind: str

    @abstractmethod
    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if the pair (z1, z2, m1, m2) should be kept."""

    def __and__(self, other: "BinRule") -> "BinRule":
        """Return the and combinator for two-point measurements."""
        return AndBinRule(bin_rules=[self, other])

    def __or__(self, other: "BinRule") -> "BinRule":
        """Return the or combinator for two-point measurements."""
        return OrBinRule(bin_rules=[self, other])

    def __invert__(self) -> "BinRule":
        """Return the inverse of the bin rule."""
        return NotBinRule(bin_rule=self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
        /,
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the BinRule class."""

        def dispatch_rule(v: Any, dispatch_handler: ValidatorFunctionWrapHandler):
            if isinstance(v, cls):
                return v
            if cls == BinRule:
                assert isinstance(v, dict)
                assert "kind" in v
                kind = v["kind"]
                concrete_cls = RULE_REGISTRY.get(kind)
                if concrete_cls is None:
                    raise ValueError(f"Unknown kind {kind}")
                return concrete_cls.model_validate(v)
            return dispatch_handler(v)

        return core_schema.no_info_wrap_validator_function(
            dispatch_rule, handler(source_type)
        )


@register_rule
class AndBinRule(BinRule):
    """Class defining the and combinator for two-point measurements."""

    kind: str = "and"
    bin_rules: list[SerializeAsAny[BinRule]]

    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if all of the bin rules pass."""
        return all(bin_rule.keep(zdist, m) for bin_rule in self.bin_rules)

    def model_post_init(self, _, /) -> None:
        """Flatten nested AndBinRules."""
        flattened = []
        for br in self.bin_rules:
            if isinstance(br, AndBinRule):
                flattened.extend(br.bin_rules)
            else:
                flattened.append(br)
        object.__setattr__(self, "bin_rules", flattened)


@register_rule
class OrBinRule(BinRule):
    """Class defining the or combinator for two-point measurements."""

    kind: str = "or"
    bin_rules: list[SerializeAsAny[BinRule]]

    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if any of the bin rules pass."""
        return any(bin_rule.keep(zdist, m) for bin_rule in self.bin_rules)

    def model_post_init(self, _, /) -> None:
        """Flatten nested OrBinRules."""
        flattened = []
        for br in self.bin_rules:
            if isinstance(br, OrBinRule):
                flattened.extend(br.bin_rules)
            else:
                flattened.append(br)
        object.__setattr__(self, "bin_rules", flattened)


@register_rule
class NotBinRule(BinRule):
    """Class defining the not combinator for two-point measurements."""

    kind: str = "not"
    bin_rule: SerializeAsAny[BinRule]

    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return the negation of keep."""
        return not self.bin_rule.keep(zdist, m)


@register_rule
class NamedBinRule(BinRule):
    """Class defining the named combinator for two-point measurements."""

    kind: str = "named"
    names: list[tuple[str, str]]

    @field_serializer("names")
    @classmethod
    def serialize_names(cls, value: list[tuple[str, str]]) -> list[list[str]]:
        """Serialize the names parameter."""
        return [list(name) for name in value]

    @field_validator("names")
    @classmethod
    def validate_names(cls, value: list[list[str]]) -> list[tuple[str, str]]:
        """Validate the names parameter."""
        for names in value:
            assert len(names) == 2
        return [(names[0], names[1]) for names in value]

    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if the pair (z1, z2) is in the list of names."""
        return (zdist[0].bin_name, zdist[1].bin_name) in self.names or (
            zdist[1].bin_name,
            zdist[0].bin_name,
        ) in self.names


@register_rule
class AutoNameBinRule(BinRule):
    """Class defining the auto-source combinator for two-point measurements."""

    kind: str = "auto-name"

    def keep(self, zdist: ZDistPair, _m: MeasurementPair) -> bool:
        """Return True if both have the same bin name."""
        return zdist[0].bin_name == zdist[1].bin_name


@register_rule
class AutoMeasurementBinRule(BinRule):
    """Class defining the auto-measurement combinator for two-point measurements."""

    kind: str = "auto-measurement"

    def keep(self, _zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if both are the same measurement."""
        return m[0] == m[1]


@register_rule
class SourceBinRule(BinRule):
    """Class defining the source combinator for two-point measurements."""

    kind: str = "source"

    def keep(self, _zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if the measurements are both source measurements."""
        return (m[0] in GALAXY_SOURCE_TYPES) and (m[1] in GALAXY_SOURCE_TYPES)


@register_rule
class LensBinRule(BinRule):
    """Class defining the lens combinator for two-point measurements."""

    kind: str = "lens"

    def keep(self, _zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if the measurements are both lens measurements."""
        return (m[0] in GALAXY_LENS_TYPES) and (m[1] in GALAXY_LENS_TYPES)


@register_rule
class FirstNeighborsBinRule(BinRule):
    """Class defining the first neighbor combinator for two-point measurements."""

    kind: str = "first-neighbors"
    num_neighbors: Annotated[int, Field(ge=0)] = 1

    def keep(self, zdist: ZDistPair, _m: MeasurementPair) -> bool:
        """Return True if the bin names are equal or one is one bin above the other."""
        pattern = re.compile(r"(?P<text>.*?)(?P<number>\d+)")
        if not (
            (match1 := pattern.match(zdist[0].bin_name))
            and (match2 := pattern.match(zdist[1].bin_name))
        ):
            return False
        return (match1["text"] == match2["text"]) and (
            abs(int(match1["number"]) - int(match2["number"])) <= self.num_neighbors
        )


@register_rule
class TypeSourceBinRule(BinRule):
    """Class defining the type-source combinator for two-point measurements."""

    kind: str = "type-source"
    type_source: TypeSource

    def keep(self, zdist: ZDistPair, _m: MeasurementPair) -> bool:
        """Return True if the bins have the same type-source."""
        return (zdist[0].type_source == zdist[1].type_source) and (
            self.type_source == zdist[0].type_source
        )
