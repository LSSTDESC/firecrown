"""Type definitions for analysis builders.

Defines enums and dataclasses used across the analysis generation system.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

import dataclasses
from enum import StrEnum
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.ccl_factory import PoweSpecAmplitudeParameter


class Frameworks(StrEnum):
    """Supported statistical analysis frameworks."""

    COBAYA = "cobaya"
    COSMOSIS = "cosmosis"
    NUMCOSMO = "numcosmo"


class FrameworkCosmology(StrEnum):
    """Cosmology required by the analysis framework."""

    NONE = "none"
    BACKGROUND = "background"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


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


@dataclasses.dataclass
class ConfigGenerator(ABC):
    """Abstract base for framework-specific configuration generators.

    Uses a builder pattern with phased state management:
    1. Create generator with output settings
    2. Add components: add_sacc(), add_factory(), add_build_parameters(), add_models()
    3. Write configuration: write_config()

    Subclasses implement write_config() to generate framework-specific files.
    """

    framework: ClassVar[Frameworks]
    output_path: Path
    prefix: str
    use_absolute_path: bool
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR
    amplitude_parameter: PoweSpecAmplitudeParameter = PoweSpecAmplitudeParameter.SIGMA8

    sacc_path: Path | None = None
    factory_source: str | Path | None = None
    build_parameters: NamedParameters | None = None
    models: list[Model] = dataclasses.field(default_factory=list)

    def add_sacc(self, sacc_path: Path) -> None:
        """Add SACC data file path to generator state.

        :param sacc_path: Path to SACC data file
        """
        self.sacc_path = sacc_path

    def add_factory(self, factory_source: Path | str) -> None:
        """Add factory file path to generator state.

        :param factory_path: Path to likelihood factory Python file
        """
        self.factory_source = factory_source

    def add_build_parameters(self, build_parameters: NamedParameters) -> None:
        """Add build parameters to generator state.

        :param build_parameters: Parameters for likelihood initialization
        """
        self.build_parameters = build_parameters

    def add_models(self, models: list[Model]) -> None:
        """Add model parameters to generator state.

        :param models: List of models with sampling parameters
        """
        self.models = models

    @abstractmethod
    def write_config(self) -> None:
        """Write framework-specific configuration files using accumulated state."""


def get_path_str(path: Path | str, use_absolute: bool) -> str:
    """Convert Path or str to appropriate path string based on use_absolute flag"""
    if isinstance(path, str):
        return path
    return path.absolute().as_posix() if use_absolute else path.name
