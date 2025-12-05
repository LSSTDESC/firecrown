"""Utility types and functions for metadata types."""

from dataclasses import dataclass
from typing import Any

from pydantic_core import core_schema

from firecrown.utils import YAMLSerializable


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


TRACER_NAMES_TOTAL = TracerNames("", "")
"""Special TracerNames instance for totals.

Represents the total/sum over all tracers, indicated by empty tracer names.
"""


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
