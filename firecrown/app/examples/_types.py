"""Type definitions for example generators.

Defines enums and dataclasses used across the example generation system.
"""

import dataclasses
from enum import StrEnum


class Frameworks(StrEnum):
    """Supported statistical analysis frameworks."""

    COBAYA = "cobaya"
    COSMOSIS = "cosmosis"
    NUMCOSMO = "numcosmo"


@dataclasses.dataclass
class Parameter:
    """Model parameter with sampling metadata.

    Defines a single parameter for cosmological or systematic modeling,
    including its prior bounds, default value, and whether it's free or fixed.
    """

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
    """Model definition with multiple parameters.

    Groups related parameters (e.g., all photo-z shifts, all galaxy biases)
    into a named model for organization in configuration files.
    """

    name: str
    parameters: list[Parameter]
    description: str
