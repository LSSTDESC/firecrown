"""Bin rule classes for combining two-point measurements."""

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

RULE_REGISTRY: dict[str, type["BinRule"]] = {}


def register_rule(cls: type["BinRule"]) -> type["BinRule"]:
    """Register a new bin rule class using its Pydantic `kind` default.

    :param cls: The BinRule class to register.

    :return: The registered BinRule class.

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
        raise ValueError(f"Duplicate rule kind {kind_value}")
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
        """Return True if the pair should be kept.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if the pair should be kept, False otherwise.
        """

    def __and__(self, other: "BinRule") -> "BinRule":
        """Return the AND combinator of this rule with another.

        :param other: Another BinRule to combine with.

        :return: An AndBinRule combining both rules.
        """
        return AndBinRule(bin_rules=[self, other])

    def __or__(self, other: "BinRule") -> "BinRule":
        """Return the OR combinator of this rule with another.

        :param other: Another BinRule to combine with.

        :return: An OrBinRule combining both rules.
        """
        return OrBinRule(bin_rules=[self, other])

    def __invert__(self) -> "BinRule":
        """Return the inverse of this bin rule.

        :return: A NotBinRule inverting this rule.
        """
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
    """Class defining the AND combinator for two-point measurements.

    This rule keeps pairs only if all contained bin rules pass.
    """

    kind: str = "and"
    bin_rules: list[SerializeAsAny[BinRule]]

    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if all of the bin rules pass.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if all rules pass, False otherwise.
        """
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
    """Class defining the OR combinator for two-point measurements.

    This rule keeps pairs if any of the contained bin rules pass.
    """

    kind: str = "or"
    bin_rules: list[SerializeAsAny[BinRule]]

    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if any of the bin rules pass.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if any rule passes, False otherwise.
        """
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
    """Class defining the NOT combinator for two-point measurements.

    This rule inverts the result of the contained bin rule.
    """

    kind: str = "not"
    bin_rule: SerializeAsAny[BinRule]

    def keep(self, zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return the negation of the contained rule's result.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects.

        :return: True if the contained rule returns False, False otherwise.
        """
        return not self.bin_rule.keep(zdist, m)


@register_rule
class NamedBinRule(BinRule):
    """Class defining the named bin rule for two-point measurements.

    This rule keeps pairs if their bin names match any of the specified name pairs.
    The order of names in the pair doesn't matter (symmetric matching).
    """

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
        """Return True if the bin name pair is in the list of names.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param m: Pair of Measurement objects (unused).

        :return: True if the bin name pair matches any configured name pair.
        """
        return (zdist[0].bin_name, zdist[1].bin_name) in self.names or (
            zdist[1].bin_name,
            zdist[0].bin_name,
        ) in self.names


@register_rule
class AutoNameBinRule(BinRule):
    """Class defining the auto-name combinator for two-point measurements.

    This rule keeps pairs of bins that have the same bin name.
    """

    kind: str = "auto-name"

    def keep(self, zdist: ZDistPair, _m: MeasurementPair) -> bool:
        """Return True if both bins have the same bin name.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if both bins have the same name, False otherwise.
        """
        return zdist[0].bin_name == zdist[1].bin_name


@register_rule
class AutoMeasurementBinRule(BinRule):
    """Class defining the auto-measurement combinator for two-point measurements.

    This rule keeps pairs where both measurements are the same.
    """

    kind: str = "auto-measurement"

    def keep(self, _zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if both measurements are the same.

        :param _zdist: Pair of InferredGalaxyZDist objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if both measurements are identical, False otherwise.
        """
        return m[0] == m[1]


@register_rule
class SourceBinRule(BinRule):
    """Class defining the source bin rule for two-point measurements.

    This rule keeps pairs where both measurements are galaxy source measurements.
    """

    kind: str = "source"

    def keep(self, _zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if both measurements are galaxy source measurements.

        :param _zdist: Pair of InferredGalaxyZDist objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if both are source measurements, False otherwise.
        """
        return (m[0] in GALAXY_SOURCE_TYPES) and (m[1] in GALAXY_SOURCE_TYPES)


@register_rule
class LensBinRule(BinRule):
    """Class defining the lens bin rule for two-point measurements.

    This rule keeps pairs where both measurements are galaxy lens measurements.
    """

    kind: str = "lens"

    def keep(self, _zdist: ZDistPair, m: MeasurementPair) -> bool:
        """Return True if both measurements are galaxy lens measurements.

        :param _zdist: Pair of InferredGalaxyZDist objects (unused).
        :param m: Pair of Measurement objects.

        :return: True if both are lens measurements, False otherwise.
        """
        return (m[0] in GALAXY_LENS_TYPES) and (m[1] in GALAXY_LENS_TYPES)


@register_rule
class FirstNeighborsBinRule(BinRule):
    """Class defining the first neighbors bin rule for two-point measurements.

    This rule keeps pairs where the bin names differ by at most num_neighbors.
    Bin names must follow the pattern <text><number>.
    """

    kind: str = "first-neighbors"
    num_neighbors: Annotated[int, Field(ge=0)] = 1

    def keep(self, zdist: ZDistPair, _m: MeasurementPair) -> bool:
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


@register_rule
class TypeSourceBinRule(BinRule):
    """Class defining the type-source bin rule for two-point measurements.

    This rule keeps pairs where both bins have the same type-source and it matches
    the configured type_source.
    """

    kind: str = "type-source"
    type_source: TypeSource

    def keep(self, zdist: ZDistPair, _m: MeasurementPair) -> bool:
        """Return True if both bins have the same matching type-source.

        :param zdist: Pair of InferredGalaxyZDist objects.
        :param _m: Pair of Measurement objects (unused).

        :return: True if both bins have matching type-sources, False otherwise.
        """
        return (zdist[0].type_source == zdist[1].type_source) and (
            self.type_source == zdist[0].type_source
        )
