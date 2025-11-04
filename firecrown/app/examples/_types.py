"""Types for example generators."""

import dataclasses
from enum import StrEnum


class Frameworks(StrEnum):
    """Supported frameworks for example generation."""

    COBAYA = "cobaya"
    COSMOSIS = "cosmosis"
    NUMCOSMO = "numcosmo"


@dataclasses.dataclass
class Parameter:
    """A parameter with associated metadata."""

    name: str
    symbol: str
    lower_bound: float
    upper_bound: float
    scale: float
    abstol: float
    default_value: float
    free: bool


@dataclasses.dataclass
class Model:
    """A model with associated parameters."""

    name: str
    parameters: list[Parameter]
    description: str
