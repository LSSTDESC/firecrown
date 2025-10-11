"""Interpolation configuration for two-point theory calculations."""

from enum import Flag, auto
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class ApplyInterpolationWhen(Flag):
    """Flags controlling when to apply interpolation of multipole moments.

    These flags specify the contexts in which interpolation should be used instead of
    computing all multipoles exactly. When `NONE` is set, all multipoles are computed
    directly. When any flag is set, only the multipoles defined in `LogLinearElls()` are
    computed exactly; the others will be interpolated.

    Note: For real-space correlation functions xi(theta), interpolation is handled
    internally by CCL and does not depend on these flags.
    """

    NONE = 0
    REAL = auto()
    HARMONIC = auto()
    HARMONIC_WINDOW = auto()
    DEFAULT = REAL | HARMONIC_WINDOW
    ALL = REAL | HARMONIC | HARMONIC_WINDOW

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Custom schema for Pydantic to support Flag (de)serialization from string."""

        def validate(v: Any) -> "ApplyInterpolationWhen":
            if isinstance(v, cls):
                return v
            if isinstance(v, str):
                parts = v.strip().split("|")
                result = cls.NONE
                for part in parts:
                    part = part.strip()
                    try:
                        result |= cls[part]
                    except KeyError as exc:
                        raise ValueError(f"Invalid flag name: '{part}'") from exc
                return result
            raise TypeError(f"Cannot parse ApplyInterpolationWhen from value: {v}")

        def serialize(v: "ApplyInterpolationWhen") -> str:
            if v == cls.NONE:
                return "NONE"
            return "|".join(
                flag.name
                for flag in cls
                if flag & v and flag != cls.NONE and flag.name is not None
            )

        return core_schema.no_info_before_validator_function(
            validate,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(serialize),
        )
