"""Enumeration types for CCL factory module."""

from enum import StrEnum, auto
from typing import Any

from pydantic_core import core_schema


class PoweSpecAmplitudeParameter(StrEnum):
    """This class defines the two-point correlation space.

    The two-point correlation space can be either real or harmonic. The real space
    corresponds measurements in terms of angular separation, while the harmonic space
    corresponds to measurements in terms of spherical harmonics decomposition.
    """

    AS = auto()
    SIGMA8 = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the PoweSpecAmplitudeParameter class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


class CCLCreationMode(StrEnum):
    """This class defines the CCL instance creation mode.

    The DEFAULT mode represents the current CCL behavior. It will use CCL's calculator
    mode if `prepare` is called with a `CCLCalculatorArgs` object. Otherwise, it will
    use the default CCL mode.

    The MU_SIGMA_ISITGR mode enables the mu-sigma modified gravity model with the ISiTGR
    transfer function, it is not compatible with the Calculator mode.

    The PURE_CCL_MODE mode will create a CCL instance with the default parameters. It is
    not compatible with the Calculator mode.
    """

    DEFAULT = auto()
    MU_SIGMA_ISITGR = auto()
    PURE_CCL_MODE = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the CCLCreationMode class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


class CCLPureModeTransferFunction(StrEnum):
    """This class defines the transfer function to use in PURE_CCL_MODE.

    The options are those available in CCL for the transfer_function argument.
    See https://ccl.readthedocs.io/en/latest/api/pyccl.cosmology.html
    """

    BBKS = auto()
    BOLTZMANN_CAMB = auto()
    BOLTZMANN_CLASS = auto()
    EISENSTEIN_HU = auto()
    EISENSTEIN_HU_NOWIGGLES = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the CCLPureModeTransferFunction class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )
