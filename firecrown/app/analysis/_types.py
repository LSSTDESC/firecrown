"""Type definitions for analysis builders.

Defines enums and dataclasses used across the analysis generation system.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

import dataclasses
from enum import StrEnum
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Sequence, Any
from pydantic import BaseModel, ConfigDict

import pyccl

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.ccl_factory import CAMBExtraParams, PoweSpecAmplitudeParameter


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
    description: str
    parameters: list[Parameter]


class PriorUniform(BaseModel):
    """Uniform prior definition."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float

    def model_post_init(self, _, /) -> None:
        """Validate that lower < upper."""
        if not self.lower < self.upper:
            raise ValueError("lower must be < upper")


class PriorGaussian(BaseModel):
    """Gaussian prior definition."""

    model_config = ConfigDict(extra="forbid")

    mean: float
    sigma: float

    def model_post_init(self, _, /) -> None:
        """Validate that sigma > 0."""
        if not self.sigma > 0:
            raise ValueError("sigma must be > 0")


Prior = PriorUniform | PriorGaussian


# This dataclass defines cosmological parameters used in CCL (Core Cosmology Library)
# with their metadata.
class CCLCosmologyParameters(BaseModel):  # pylint: disable=too-many-instance-attributes
    """CCL cosmological parameter with metadata.

    Defines a cosmological parameter used in CCL with its name,
    symbol for display, and default value.
    """

    model_config = ConfigDict(extra="forbid")

    Omega_c: float | None = None
    Omega_b: float | None = None
    h: float | None = None
    n_s: float | None = None
    sigma8: float | None = None
    A_s: float | None = None
    Omega_k: float = 0.0
    Omega_g: float | None = None
    Neff: float | None = None
    m_nu: float | Sequence[float] = 0.0
    mass_split: str = "normal"

    w0: float = -1.0
    wa: float = 0.0

    T_CMB: float = 2.7255
    T_ncdm: float = 0.71611

    transfer_function: str | Any = "boltzmann_camb"
    matter_power_spectrum: str | Any = "halofit"

    baryonic_effects: Any | None = None
    mg_parametrization: Any | None = None

    extra_parameters: CAMBExtraParams | None = None

    def to_ccl_cosmology(self) -> pyccl.Cosmology:
        """Convert to CCL cosmology dictionary.

        :return: Dictionary of cosmological parameters for CCL
        """
        return pyccl.Cosmology(**self.model_dump(exclude_none=True))


class CCLCosmologyPriors(BaseModel):
    """Priors for CCL cosmological parameters.

    Defines priors for cosmological parameters used in CCL.
    """

    model_config = ConfigDict(extra="forbid")

    Omega_c: Prior | None = None
    Omega_b: Prior | None = None
    h: Prior | None = None
    n_s: Prior | None = None
    sigma8: Prior | None = None
    A_s: Prior | None = None
    Omega_k: Prior | None = None
    Neff: Prior | None = None
    m_nu: Prior | None = None

    w0: Prior | None = None
    wa: Prior | None = None

    def is_empty(self) -> bool:
        """Check if no priors are defined.

        :return: True if no priors are set, False otherwise
        """
        fields = CCLCosmologyPriors.model_fields
        return all(getattr(self, field) is None for field in fields.keys())


class CCLCosmologyAnalysisSpec(BaseModel):
    """CCL cosmology analysis specification.

    Defines the cosmology and priors for CCL cosmology analysis.
    """

    model_config = ConfigDict(extra="forbid")

    cosmology: CCLCosmologyParameters
    priors: CCLCosmologyPriors

    def _validate_as_sigma8_exclusivity(self) -> None:
        """Validate that either sigma8 or A_s is set, but not both."""
        A_s_val = self.cosmology.A_s
        sig8_val = self.cosmology.sigma8
        A_s_prior = self.priors.A_s
        sig8_prior = self.priors.sigma8
        # Exactly one of A_s and sigma8 must have a value
        if (A_s_val is None) == (sig8_val is None):
            raise ValueError(
                f"Exactly one of A_s and sigma8 must be supplied. "
                f"(A_s={A_s_val}, sigma8={sig8_val})"
            )
        # Exactly one of the priors must be provided
        if (A_s_prior is not None) and (sig8_prior is not None):
            raise ValueError(
                f"Both A_s and sigma8 priors cannot be provided. "
                f"(A_s_prior={A_s_prior}, sigma8_prior={sig8_prior})"
            )
        # The chosen param must match the chosen prior
        if A_s_prior is not None and A_s_val is None:
            raise ValueError("A_s prior provided but A_s value is missing.")
        if sig8_prior is not None and sig8_val is None:
            raise ValueError("sigma8 prior provided but sigma8 value is missing.")

    def _validate_m_nu_priors(self) -> None:
        """Validate that m_nu priors match the number of masses."""
        m_nu_val = self.cosmology.m_nu
        m_nu_priors = self.priors.m_nu
        if isinstance(m_nu_val, float):
            num_masses = 1 if m_nu_val > 0 else 0
        else:
            num_masses = len(m_nu_val)
        if m_nu_priors is not None and num_masses == 0:
            raise ValueError("m_nu prior provided but no massive neutrinos defined.")

    def model_post_init(self, _, /) -> None:
        """Run validations after initialization."""
        self._validate_as_sigma8_exclusivity()
        self._validate_m_nu_priors()

    @classmethod
    def vanilla_lcdm(cls) -> "CCLCosmologyAnalysisSpec":
        """Create a vanilla LCDM cosmology analysis spec with standard parameters."""
        cosmology = CCLCosmologyParameters(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.67,
            n_s=0.96,
            sigma8=0.81,
        )
        priors = CCLCosmologyPriors(
            Omega_c=PriorUniform(lower=0.06, upper=0.46),
            Omega_b=PriorUniform(lower=0.03, upper=0.07),
            sigma8=PriorUniform(lower=0.7, upper=1.2),
        )
        return cls(cosmology=cosmology, priors=priors)

    def get_num_massive_neutrinos(self) -> int:
        """Get the number of massive neutrinos defined in the cosmology."""
        m_nu_val = self.cosmology.m_nu
        if isinstance(m_nu_val, float):
            return 1 if m_nu_val > 0 else 0
        return len(m_nu_val)

    def get_amplitude_parameter(self) -> PoweSpecAmplitudeParameter:
        """Get the amplitude parameter for the power spectrum."""
        if self.cosmology.sigma8 is not None:
            return PoweSpecAmplitudeParameter.SIGMA8
        return PoweSpecAmplitudeParameter.SIGMA8


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
    cosmo_spec: CCLCosmologyAnalysisSpec
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR

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
    """Convert Path or str to appropriate path string based on use_absolute flag."""
    if isinstance(path, str):
        return path
    return path.absolute().as_posix() if use_absolute else path.name
