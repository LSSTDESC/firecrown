"""Bin pair selector classes for filtering tomographic bin pairs.

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
from typing import Any

from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    SerializeAsAny,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
)
from pydantic_core import PydanticUndefined, core_schema

from firecrown.metadata_types._two_point_tracers import TomographicBin
from firecrown.metadata_types._measurements import (
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
    Measurement,
)
from firecrown.metadata_types._utils import TypeSource

BIN_PAIR_SELECTOR_REGISTRY: dict[str, type["BinPairSelector"]] = {}


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
    if kind_value in BIN_PAIR_SELECTOR_REGISTRY:
        raise ValueError(f"Duplicate pair selector kind {kind_value}")
    BIN_PAIR_SELECTOR_REGISTRY[kind_value] = cls
    return cls


# Type aliases for clarity
TomographicBinPair = tuple[TomographicBin, TomographicBin]
"""A pair of tomographic bin distributions to be correlated."""

MeasurementPair = tuple[Measurement, Measurement]
"""A pair of measurement types (e.g., galaxy_shear_plus, galaxy_density)."""


class BinPairSelector(BaseModel):
    """Base class for filtering pairs of tomographic bins in two-point measurements.

    A BinPairSelector determines which pairs of `TomographicBin` bins should be
    included when constructing `TwoPointXY` objects. Concrete implementations define
    specific selection criteria (e.g., auto-correlations only, specific bin names,
    measurement types, etc.).

    Selectors support logical composition via operators:
    - AND: selector1 & selector2
    - OR: selector1 | selector2
    - NOT: ~selector
    """

    kind: str

    @abstractmethod
    def keep(self, zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if the pair should be kept.

        :param zdist: Pair of TomographicBin objects.
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
                concrete_cls = BIN_PAIR_SELECTOR_REGISTRY.get(kind)
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

        :param zdist: Pair of TomographicBin objects.
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

        :param zdist: Pair of TomographicBin objects.
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

        :param zdist: Pair of TomographicBin objects.
        :param m: Pair of Measurement objects.

        :return: True if the contained pair selector returns False, False otherwise.
        """
        return not self.pair_selector.keep(zdist, m)


class BadSelector(BinPairSelector):
    """A BinPairSelector that always raises NotImplementedError."""

    kind: str = "bad-selector"

    def keep(self, _zdist: TomographicBinPair, _m: MeasurementPair) -> bool:
        """Raise NotImplementedError always.

        :raise NotImplementedError: Always raised since this selector should not be
            used.
        """
        raise NotImplementedError("BadSelector should not be used directly.")


class CompositeSelector(BinPairSelector):
    """Base class for selectors composed from other selectors."""

    _impl: BinPairSelector = BadSelector()

    def keep(self, zdist: TomographicBinPair, m: MeasurementPair):
        """Delegate to the underlying selector implementation."""
        assert isinstance(self._impl, BinPairSelector)
        return self._impl.keep(zdist, m)


@register_bin_pair_selector
class NamedBinPairSelector(BinPairSelector):
    """Selector for explicitly specified bin name pairs.

    This selector keeps bin pairs if their names match any entry in the configured
    list exactly, in the given order. The matching is order-dependent: (name1, name2)
    is different from (name2, name1).

    Example:
        names=[("bin0", "bin1"), ("bin2", "bin2")]  # Matches in that exact order
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

        :param zdist: Pair of TomographicBin objects.
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

        :param zdist: Pair of TomographicBin objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if both bins have the same name, False otherwise.
        """
        return zdist[0].bin_name == zdist[1].bin_name


@register_bin_pair_selector
class CrossNameBinPairSelector(CompositeSelector):
    """Selector for cross-correlations: excludes auto-name pairs.

    This selector keeps only pairs where both bins have different names.
    """

    kind: str = "cross-name"

    def model_post_init(self, _: Any, /) -> None:
        """Invert AutoNameBinPairSelector."""
        self._impl = ~AutoNameBinPairSelector()


@register_bin_pair_selector
class AutoMeasurementBinPairSelector(BinPairSelector):
    """Selector for auto-correlations based on measurement type.

    This selector keeps only pairs where both measurements are identical
    (e.g., both are galaxy_shear_plus or both are galaxy_density).
    """

    kind: str = "auto-measurement"

    def keep(self, _zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if both measurements are the same.

        :param _zdist: Pair of TomographicBin objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if both measurements are identical, False otherwise.
        """
        return m[0] == m[1]


@register_bin_pair_selector
class CrossMeasurementBinPairSelector(CompositeSelector):
    """Selector for cross-correlations: excludes auto-measurement pairs.

    This selector keeps only pairs where both measurements differ.
    """

    kind: str = "cross-measurement"

    def model_post_init(self, _: Any, /) -> None:
        """Invert AutoMeasurementBinPairSelector."""
        self._impl = ~AutoMeasurementBinPairSelector()


@register_bin_pair_selector
class AutoBinPairSelector(CompositeSelector):
    """Selector for auto-correlations based on both bin name and measurement type.

    This selector keeps only pairs where both bins have identical names and both
    measurements are identical, effectively selecting auto-correlations within the same
    tomographic bin and measurement type.
    """

    kind: str = "auto-bin"

    def model_post_init(self, _: Any, /) -> None:
        """Initialize as composite of auto-name and auto-measurement selectors."""
        self._impl = AutoNameBinPairSelector() & AutoMeasurementBinPairSelector()


@register_bin_pair_selector
class CrossBinPairSelector(CompositeSelector):
    """Selector for cross-correlations: excludes auto-bin pairs.

    This selector keeps only pairs where the bin names differ, effectively selecting
    cross-correlations between different tomographic bins.
    """

    kind: str = "cross-bin"

    def model_post_init(self, _: Any, /) -> None:
        """Invert AutoBinPairSelector."""
        self._impl = ~AutoBinPairSelector()


@register_bin_pair_selector
class LeftMeasurementBinPairSelector(BinPairSelector):
    """Selector filtering pairs by the left (x) measurement type.

    This selector keeps pairs where the left (x) measurement is one of the
    configured measurement types.

    Example:
        # Select pairs where left measurement is a source (shear) measurement
        selector = LeftMeasurementBinPairSelector(measurement_set=GALAXY_SOURCE_TYPES)
    """

    kind: str = "left-measurement"
    measurement_set: set[Measurement]

    def keep(self, _zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if the left measurement is in the configured set.

        :param _zdist: Pair of TomographicBin objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if the left measurement is in the set, False otherwise.
        """
        return m[0] in self.measurement_set


@register_bin_pair_selector
class RightMeasurementBinPairSelector(BinPairSelector):
    """Selector filtering pairs by the right (y) measurement type.

    This selector keeps pairs where the right (y) measurement is one of the
    configured measurement types.

    Example:
        # Select pairs where right measurement is a lens (density) measurement
        selector = RightMeasurementBinPairSelector(measurement_set=GALAXY_LENS_TYPES)
    """

    kind: str = "right-measurement"
    measurement_set: set[Measurement]

    def keep(self, _zdist: TomographicBinPair, m: MeasurementPair) -> bool:
        """Return True if the right measurement is in the configured set.

        :param _zdist: Pair of TomographicBin objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if the right measurement is in the set, False otherwise.
        """
        return m[1] in self.measurement_set


@register_bin_pair_selector
class SourceBinPairSelector(CompositeSelector):
    """Selector for galaxy source (weak lensing) measurement pairs.

    This selector keeps pairs where both measurements are galaxy source types,
    which correspond to weak lensing shear measurements (e.g., galaxy_shear_plus,
    galaxy_shear_minus).

    Implementation note: This is a composite selector combining
    LeftMeasurementBinPairSelector and RightMeasurementBinPairSelector
    with GALAXY_SOURCE_TYPES.
    """

    kind: str = "source"

    def model_post_init(self, _: Any, /) -> None:
        """Initialize as composite of left and right source measurement selectors."""
        self._impl = LeftMeasurementBinPairSelector(
            measurement_set=set(GALAXY_SOURCE_TYPES)
        ) & RightMeasurementBinPairSelector(measurement_set=set(GALAXY_SOURCE_TYPES))


@register_bin_pair_selector
class LensBinPairSelector(CompositeSelector):
    """Selector for galaxy lens (clustering) measurement pairs.

    This selector keeps pairs where both measurements are galaxy lens types,
    which correspond to galaxy number density or clustering measurements.

    Implementation note: This is a composite selector combining
    LeftMeasurementBinPairSelector and RightMeasurementBinPairSelector
    with GALAXY_LENS_TYPES.
    """

    kind: str = "lens"

    def model_post_init(self, _: Any, /) -> None:
        """Initialize as composite of left and right lens measurement selectors."""
        self._impl = LeftMeasurementBinPairSelector(
            measurement_set=set(GALAXY_LENS_TYPES)
        ) & RightMeasurementBinPairSelector(measurement_set=set(GALAXY_LENS_TYPES))


@register_bin_pair_selector
class SourceLensBinPairSelector(CompositeSelector):
    """Selector for mixed source-lens measurement pairs.

    This selector keeps pairs where the left measurement is a galaxy source type
    (weak lensing shear) and the right measurement is a galaxy lens type
    (density/clustering). The measurement order follows the convention: source types
    are ordered before lens types.

    Implementation note: This is a composite selector combining
    LeftMeasurementBinPairSelector with GALAXY_SOURCE_TYPES and
    RightMeasurementBinPairSelector with GALAXY_LENS_TYPES.
    """

    kind: str = "source-lens"

    def model_post_init(self, _: Any, /) -> None:
        """Initialize as composite of source-left and lens-right selectors."""
        self._impl = LeftMeasurementBinPairSelector(
            measurement_set=set(GALAXY_SOURCE_TYPES)
        ) & RightMeasurementBinPairSelector(measurement_set=set(GALAXY_LENS_TYPES))


@register_bin_pair_selector
class NameDiffBinPairSelector(BinPairSelector):
    """Selector for tomographic bins based on numeric index proximity.

    This selector filters bin pairs based on the numeric suffix of their names,
    with optional constraints on the text prefix. Bin names must follow the pattern
    `<text><number>`, where `<text>` is any text prefix and `<number>` is the bin
    index suffix.

    The selector keeps pairs where the numeric indices differ by one of the allowed
    values in `neighbors_diff`. If `same_name_prefix=True`, the text prefixes must
    also be identical. If `same_name_prefix=False`, the text prefixes must differ.

    :param same_name_prefix: If True, keep pairs with identical text prefixes;
        if False, keep pairs with different text prefixes.
    :param neighbors_diff: An integer or list of integers specifying which index
        differences are allowed. Differences are computed as (left_index -
        right_index).

    Example:
        With same_name_prefix=True, neighbors_diff=1 (auto-pairs with adjacent
        indices):

        - Keeps: ("bin0", "bin1"), ("bin1", "bin0"), ("bin2", "bin2")
        - Rejects: ("bin0", "bin2"), ("bin0", "src0")

        With same_name_prefix=False, neighbors_diff=[1, -1] (cross-pairs with different
        prefixes):

        - Keeps: ("src0", "bin1"), ("bin1", "src0")
        - Rejects: ("bin0", "bin1"), ("src0", "src1")
    """

    kind: str = "name-diff"
    same_name_prefix: bool = True
    neighbors_diff: int | list[int] = 1

    def keep(self, zdist: TomographicBinPair, _m: MeasurementPair) -> bool:
        """Return True if bin name indices differ by an allowed amount.

        Both bin names must match the pattern <text><number>. The numeric parts are
        extracted and their difference is checked against the allowed values.
        If same_name_prefix is set, the text parts must also satisfy the constraint.

        :param zdist: Pair of TomographicBin objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if bins are neighbors, False otherwise.
        """
        pattern = re.compile(r"^(?P<text>.*?)(?P<number>\d+)$")
        allowed_neighbors = (
            [self.neighbors_diff]
            if isinstance(self.neighbors_diff, int)
            else self.neighbors_diff
        )
        if not (
            (match1 := pattern.match(zdist[0].bin_name))
            and (match2 := pattern.match(zdist[1].bin_name))
        ):
            return False
        if self.same_name_prefix and (match1["text"] != match2["text"]):
            return False
        if not self.same_name_prefix and (match1["text"] == match2["text"]):
            return False

        return (int(match1["number"]) - int(match2["number"])) in allowed_neighbors


@register_bin_pair_selector
class AutoNameDiffBinPairSelector(CompositeSelector):
    """Selector for tomographic bins with identical prefix and nearby indices.

    This selector keeps bin pairs where:
    1. Both bins have identical text prefixes (e.g., both start with "bin")
    2. Their numeric indices differ by one of the allowed amounts in `neighbors_diff`

    This is useful for selecting bin combinations with similar properties while
    allowing for spatial or redshift proximity constraints.

    :param neighbors_diff: An integer or list of integers specifying which index
        differences are allowed. For example, [0, 1, -1] allows identical
        indices and immediate neighbors.

    Example:
        With neighbors_diff=1 (same prefix, adjacent or identical indices):
        - Keeps: ("bin0", "bin1"), ("bin1", "bin0"), ("bin2", "bin2")
        - Rejects: ("bin0", "bin2"), ("src0", "bin0")
    """

    kind: str = "auto-name-diff"

    neighbors_diff: int | list[int] = 1

    def model_post_init(self, _: Any, /) -> None:
        """Initialize as NameDiffBinPairSelector with same_name_prefix=True."""
        self._impl = NameDiffBinPairSelector(
            same_name_prefix=True, neighbors_diff=self.neighbors_diff
        )


@register_bin_pair_selector
class CrossNameDiffBinPairSelector(CompositeSelector):
    """Selector for tomographic bins with different prefixes and nearby indices.

    This selector keeps bin pairs where:
    1. Both bins have different text prefixes (e.g., one "bin" and one "src")
    2. Their numeric indices differ by one of the allowed amounts in `neighbors_diff`

    This is useful for selecting cross-type bin combinations (e.g., clustering-lensing
    pairs) with spatial or redshift proximity constraints.

    :param neighbors_diff: An integer or list of integers specifying which index
        differences are allowed.

    Example:
        With neighbors_diff=[0, 1, -1] (different prefixes, nearby indices):
        - Keeps: ("src0", "bin0"), ("src0", "bin1"), ("bin1", "src0")
        - Rejects: ("bin0", "bin1"), ("src0", "src1")
    """

    kind: str = "cross-name-diff"

    neighbors_diff: int | list[int] = 1

    def model_post_init(self, _: Any, /) -> None:
        """Initialize as NameDiffBinPairSelector with same_name_prefix=False."""
        self._impl = NameDiffBinPairSelector(
            same_name_prefix=False, neighbors_diff=self.neighbors_diff
        )


@register_bin_pair_selector
class ThreeTwoBinPairSelector(CompositeSelector):
    """Selector for 3x2pt analysis bin pairs.

    This selector implements the standard 3x2pt cosmological analysis selection,
    which includes three types of two-point correlations:

    1. Source-source (cosmic shear): auto-correlations of weak lensing shear
    2. Lens-lens (galaxy clustering): auto-correlations of galaxy positions
    3. Source-lens (galaxy-galaxy lensing): cross-correlations between source and lens
       samples with different bin names

    The galaxy-galaxy lensing component explicitly excludes auto-correlations
    (same bin name) to avoid mixing source and lens samples from the same
    tomographic bin, which is the standard practice in 3x2pt analyses.

    Implementation note: This is a composite selector combining:
        SourceBinPairSelector() | LensBinPairSelector() |
        (SourceLensBinPairSelector() & CrossNameBinPairSelector())

    Example:
        selector = ThreeTwoBinPairSelector()
        # Includes: source-source, lens-lens, and cross-name source-lens pairs

    :param source_dist: Maximum allowed index separation (left minus right) for
        source-source bin pairs. The allowed set is ``range(-source_dist,
        source_dist)`` (zero included, upper bound exclusive).
    :param lens_dist: Maximum allowed index separation (left minus right) for
        lens-lens bin pairs. The allowed set is ``range(-lens_dist, lens_dist)``.
    :param source_lens_dist: Maximum positive index separation allowed for
        source-lens pairs. The allowed set is ``range(1, source_lens_dist + 1)``.
        Only pairs with different prefixes (enforced by ``CrossNameDiff``) are kept.
    """

    kind: str = "3x2pt"

    source_dist: int = 5
    lens_dist: int = 5
    source_lens_dist: int = 5

    def model_post_init(self, _: Any, /) -> None:
        """Initialize as composite of source, lens, and source-lens selectors."""
        self._impl = (
            (
                SourceBinPairSelector()
                & AutoNameDiffBinPairSelector(
                    neighbors_diff=list(range(-self.source_dist, self.source_dist))
                )
            )
            | (
                LensBinPairSelector()
                & AutoNameDiffBinPairSelector(
                    neighbors_diff=list(range(-self.lens_dist, self.lens_dist))
                )
            )
            | (
                SourceLensBinPairSelector()
                & CrossNameDiffBinPairSelector(
                    neighbors_diff=list(range(1, self.source_lens_dist + 1))
                )
            )
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

        :param zdist: Pair of TomographicBin objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if both bins have matching type-sources, False otherwise.
        """
        return (zdist[0].type_source == zdist[1].type_source) and (
            self.type_source == zdist[0].type_source
        )
