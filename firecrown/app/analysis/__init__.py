"""Analysis building infrastructure for Firecrown.

Provides base classes and utilities for building complete analysis setups
with data files, factory files, and framework-specific configurations.

Public API (backward compatibility guaranteed):
    - AnalysisBuilder: Base class for building analyses
    - Frameworks: Enum of supported frameworks
    - Model: Model parameter definition
    - Parameter: Individual parameter definition

Internal modules (prefixed with _) are implementation details and may change
without notice. Only use the public API exported in __all__.
"""

from ._analysis_builder import AnalysisBuilder
from ._types import (
    Frameworks,
    Model,
    Parameter,
    ConfigGenerator,
    FrameworkCosmology,
    PriorUniform,
    PriorGaussian,
    CCLCosmologySpec,
    COSMO_DESC,
)
from ._config_generator import (
    get_generator,
    CosmosisConfigGenerator,
    NumCosmoConfigGenerator,
    CobayaConfigGenerator,
)
from ._download import download_from_url, copy_template

__all__ = [
    "AnalysisBuilder",
    "Frameworks",
    "Model",
    "Parameter",
    "PriorUniform",
    "PriorGaussian",
    "CCLCosmologySpec",
    "COSMO_DESC",
    "get_generator",
    "ConfigGenerator",
    "CosmosisConfigGenerator",
    "NumCosmoConfigGenerator",
    "CobayaConfigGenerator",
    "FrameworkCosmology",
    "download_from_url",
    "copy_template",
]
