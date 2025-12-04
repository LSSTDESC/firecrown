"""Bin pair selector classes for filtering tomographic bin pairs in two-point measurements.

This module provides a flexible system for selecting which pairs of tomographic bins
should be included when constructing two-point correlation functions. Selectors can be
combined using logical operators (AND, OR, NOT) to create complex selection rules.

Example usage:
    # Select only auto-correlations with the same bin name
    selector = AutoNameBinPairSelector() & AutoMeasurementBinPairSelector()

    # Select source pairs or lens pairs
    selector = SourceBinPairSelector() | LensBinPairSelector()

    # Select everything except auto-correlations
    selector = ~AutoNameBinPairSelector()
"""

import re
from abc import abstractmethod
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    SerializeAsAny,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
)
from pydantic_core import PydanticUndefined, core_schema

from firecrown.metadata_types._inferred_galaxy_zdist import InferredGalaxyZDist
from firecrown.metadata_types._measurements import (
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
    Measurement,
)
from firecrown.metadata_types._utils import TypeSource

RULE_REGISTRY: dict[str, type["BinPairSelector"]] = {}


def register_bin_pair_selector(cls: type["BinPairSelector"]) -> type["BinPairSelector"]:
    """Register a new bin pair selector class in the global registry.

    This decorator registers a BinPairSelector subclass using its Pydantic `kind`
    field default value as the registry key. This enables polymorphic deserialization
    from configuration files.

    :param cls: The BinPairSelector class to register.

    :return: The registered BinPairSelector class (for use as a decorator).

    :raises ValueError: If the class has no default for 'kind' or if the kind is
        already registered.
    """
    assert issubclass(cls, BaseModel)
    kind_field = cls.model_fields.get("kind")
    assert kind_field is not None
    if kind_field.default is PydanticUndefined:
        raise ValueError(f"{cls.__name__} has no default for 'kind'")
    kind_value = kind_field.default
    if kind_value in RULE_REGISTRY:
        raise ValueError(f"Duplicate pair selector kind {kind_value}")
    RULE_REGISTRY[kind_value] = cls
    return cls


# Type aliases for clarity
TomographicBinPair = tuple[InferredGalaxyZDist, InferredGalaxyZDist]
"""A pair of tomographic bin distributions to be correlated."""

MeasurementPair = tuple[Measurement, Measurement]
"""A pair of measurement types (e.g., galaxy_shear_plus, galaxy_density)."""


class BinPairSelector(BaseModel):
    """Base class for filtering pairs of tomographic bins in two-point measurements.

    A BinPairSelector determines which pairs of `InferredGalaxyZDist` bins should
    be included when constructing `TwoPointXY` objects. Concrete implementations
    define specific selection criteria (e.g., auto-correlations only, specific
    bin names, measurement types, etc.).

    Selectors support logical composition via operators:
    - AND: selector1 & selector2
    - OR: selector1 | selector2
    - NOT: ~selector
    """

    kind: str

    @abstractmethod
    def keep(self, zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if the pair should be kept.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if the pair should be kept, False otherwise.
        """

    def __and__(self, other: "BinPairSelector") -> "BinPairSelector":
        """Return the AND combinator of this pair selector with another.

        Example:
            # Select auto-correlations that are also source measurements
            selector = AutoNameBinPairSelector() & SourceBinPairSelector()

        :param other: Another BinPairSelector to combine with.

        :return: An AndBinPairSelector combining both pair selectors.
        """
        return AndBinPairSelector(pair_selectors=[self, other])

    def __or__(self, other: "BinPairSelector") -> "BinPairSelector":
        """Return the OR combinator of this pair selector with another.

        Example:
            # Select either source or lens pairs
            selector = SourceBinPairSelector() | LensBinPairSelector()

        :param other: Another BinPairSelector to combine with.

        :return: An OrBinPairSelector combining both pair selectors.
        """
        return OrBinPairSelector(pair_selectors=[self, other])

    def __invert__(self) -> "BinPairSelector":
        """Return the inverse of this bin pair selector.

        Example:
            # Select everything except auto-correlations
            selector = ~AutoNameBinPairSelector()

        :return: A NotBinPairSelector inverting this pair selector.
        """
        return NotBinPairSelector(pair_selector=self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
        /,
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the BinPairSelector class."""

        def dispatch_pair_selector(
            v: Any, dispatch_handler: ValidatorFunctionWrapHandler
        ):
            if isinstance(v, cls):
                return v
            if cls == BinPairSelector:
                assert isinstance(v, dict)
                assert "kind" in v
                kind = v["kind"]
                concrete_cls = RULE_REGISTRY.get(kind)
                if concrete_cls is None:
                    raise ValueError(f"Unknown kind {kind}")
                return concrete_cls.model_validate(v)
            return dispatch_handler(v)

        return core_schema.no_info_wrap_validator_function(
            dispatch_pair_selector, handler(source_type)
        )


@register_bin_pair_selector
class AndBinPairSelector(BinPairSelector):
    """Logical AND combinator for bin pair selectors.

    This selector keeps a bin pair only if ALL contained selectors accept it.
    Nested AndBinPairSelectors are automatically flattened for efficiency.
    """

    kind: str = "and"
    pair_selectors: list[SerializeAsAny[BinPairSelector]]

    def keep(self, zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if all of the bin pair selectors pass.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if all bin pair selectors pass, False otherwise.
        """
        return all(
            pair_selector.keep(zdist, m) for pair_selector in self.pair_selectors
        )

    def model_post_init(self, _, /) -> None:
        """Flatten nested AndBinPairSelectors for efficiency.

        This optimization reduces (A & B) & C to a single AndBinPairSelector
        containing [A, B, C], avoiding unnecessary nesting.
        """
        flattened = []
        for br in self.pair_selectors:
            if isinstance(br, AndBinPairSelector):
                flattened.extend(br.pair_selectors)
            else:
                flattened.append(br)
        object.__setattr__(self, "pair_selectors", flattened)


@register_bin_pair_selector
class OrBinPairSelector(BinPairSelector):
    """Logical OR combinator for bin pair selectors.

    This selector keeps a bin pair if ANY of the contained selectors accept it.
    Nested OrBinPairSelectors are automatically flattened for efficiency.
    """

    kind: str = "or"
    pair_selectors: list[SerializeAsAny[BinPairSelector]]

    def keep(self, zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if any of the bin pair selectors pass.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if any bin pair selector passes, False otherwise.
        """
        return any(
            pair_selector.keep(zdist, m) for pair_selector in self.pair_selectors
        )

    def model_post_init(self, _, /) -> None:
        """Flatten nested OrBinPairSelectors for efficiency.

        This optimization reduces (A | B) | C to a single OrBinPairSelector
        containing [A, B, C], avoiding unnecessary nesting.
        """
        flattened = []
        for br in self.pair_selectors:
            if isinstance(br, OrBinPairSelector):
                flattened.extend(br.pair_selectors)
            else:
                flattened.append(br)
        object.__setattr__(self, "pair_selectors", flattened)


@register_bin_pair_selector
class NotBinPairSelector(BinPairSelector):
    """Logical NOT combinator for bin pair selectors.

    This selector inverts the result of the contained selector, accepting pairs
    that would be rejected and vice versa.
    """

    kind: str = "not"
    pair_selector: SerializeAsAny[BinPairSelector]

    def keep(self, zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return the negation of the contained pair selector's result.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if the contained pair selector returns False, False otherwise.
        """
        return not self.pair_selector.keep(zdist, m)


@register_bin_pair_selector
class NamedBinPairSelector(BinPairSelector):
    """Selector for explicitly specified bin name pairs.

    This selector keeps bin pairs if their names match any entry in the configured
    list. Matching is symmetric: (name1, name2) is considered equivalent to
    (name2, name1).

    Example:
        names=[("bin0", "bin1"), ("bin2", "bin2")]  # Cross-correlation and one auto
    """

    kind: str = "named"
    names: list[tuple[str, str]]

    @field_serializer("names")
    @classmethod
    def serialize_names(cls, value: list[tuple[str, str]]) -> list[list[str]]:
        """Serialize name tuples to lists for JSON/YAML compatibility.

        :param value: List of name tuples.
        :return: List of name lists (tuples converted to lists).
        """
        return [list(name) for name in value]

    @field_validator("names")
    @classmethod
    def validate_names(cls, value: list[list[str]]) -> list[tuple[str, str]]:
        """Validate and convert name lists to tuples.

        :param value: List of name lists (from deserialization).
        :return: List of name tuples.
        :raises AssertionError: If any name list doesn't have exactly 2 elements.
        """
        for names in value:
            assert len(names) == 2
        return [(names[0], names[1]) for names in value]

    def keep(self, zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if the bin name pair matches any configured pair.

        Note: Matching is currently order-dependent. To achieve symmetric matching,
        include both (name1, name2) and (name2, name1) in the names list.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects (unused).

        :return: True if the bin name pair matches any configured name pair.
        """
        return (zdist[0].bin_name, zdist[1].bin_name) in self.names


@register_bin_pair_selector
class AutoNameBinPairSelector(BinPairSelector):
    """Selector for auto-correlations based on bin names.

    This selector keeps only pairs where both bins have identical names,
    effectively selecting auto-correlations within the same tomographic bin.
    """

    kind: str = "auto-name"

    def keep(self, zdist: TomographicBinPair, _m: MeasurementPair) -> bool:
        """Return True if both bins have the same bin name.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if both bins have the same name, False otherwise.
        """
        return zdist[0].bin_name == zdist[1].bin_name


@register_bin_pair_selector
class AutoMeasurementBinPairSelector(BinPairSelector):
    """Selector for auto-correlations based on measurement type.

    This selector keeps only pairs where both measurements are identical
    (e.g., both are galaxy_shear_plus or both are galaxy_density).
    """

    kind: str = "auto-measurement"

    def keep(self, _zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if both measurements are the same.

        :param _zdist: Pair of InferredGalaxyZDist objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if both measurements are identical, False otherwise.
        """
        return m[0] == m[1]


@register_bin_pair_selector
class SourceBinPairSelector(BinPairSelector):
    """Selector for galaxy source (weak lensing) measurement pairs.

    This selector keeps pairs where both measurements are galaxy source types,
    which correspond to weak lensing shear measurements (e.g., galaxy_shear_plus,
    galaxy_shear_minus).
    """

    kind: str = "source"

    def keep(self, _zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if both measurements are galaxy source measurements.

        :param _zdist: Pair of InferredGalaxyZDist objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if both are source measurements, False otherwise.
        """
        return (m[0] in GALAXY_SOURCE_TYPES) and (m[1] in GALAXY_SOURCE_TYPES)


@register_bin_pair_selector
class LensBinPairSelector(BinPairSelector):
    """Selector for galaxy lens (clustering) measurement pairs.

    This selector keeps pairs where both measurements are galaxy lens types,
    which correspond to galaxy number density or clustering measurements.
    """

    kind: str = "lens"

    def keep(self, _zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if both measurements are galaxy lens measurements.

        :param _zdist: Pair of InferredGalaxyZDist objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if both are lens measurements, False otherwise.
        """
        return (m[0] in GALAXY_LENS_TYPES) and (m[1] in GALAXY_LENS_TYPES)


@register_bin_pair_selector
class FirstNeighborsBinPairSelector(BinPairSelector):
    """Selector for neighboring tomographic bins.

    This selector keeps bin pairs where the bin indices differ by at most
    `num_neighbors`. Bin names must follow the pattern <text><number>, where
    the text part must be identical and the numeric part represents the bin index.

    Example:
        With num_neighbors=1:
        - Keeps: ("bin0", "bin1"), ("bin2", "bin2"), ("src1", "src2")
        - Rejects: ("bin0", "bin2"), ("bin0", "src0")
    """

    kind: str = "first-neighbors"
    num_neighbors: Annotated[int, Field(ge=0)] = 1

    def keep(self, zdist: TomographicBinPair, _m: MeasurementPair) -> bool:
        """Return True if bin names differ by at most num_neighbors.

        Bin names must match the pattern <text><number>. The text parts must
        be identical, and the numeric parts must differ by at most num_neighbors.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if bins are neighbors, False otherwise.
        """
        pattern = re.compile(r"^(?P<text>.*?)(?P<number>\d+)$")
        if not (
            (match1 := pattern.match(zdist[0].bin_name))
            and (match2 := pattern.match(zdist[1].bin_name))
        ):
            return False
        return (match1["text"] == match2["text"]) and (
            abs(int(match1["number"]) - int(match2["number"])) <= self.num_neighbors
        )


@register_bin_pair_selector
class TypeSourceBinPairSelector(BinPairSelector):
    """Selector for bins with matching type-source.

    This selector keeps bin pairs where both bins have identical type-sources
    that also match the configured value. The type-source typically indicates
    whether data comes from spectroscopy or photometry.
    """

    kind: str = "type-source"
    type_source: TypeSource

    def keep(self, zdist: TomographicBinPair, _m: MeasurementPair) -> bool:
        """Return True if both bins have the same matching type-source.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if both bins have matching type-sources, False otherwise.
        """
        return (zdist[0].type_source == zdist[1].type_source) and (
            self.type_source == zdist[0].type_source
        )
