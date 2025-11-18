"""Type definitions for framework configuration generators.

Provides base classes, enums, and data models for generating framework-specific
configurations. Includes cosmology specifications, parameter definitions, and
the abstract ConfigGenerator base class.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

import dataclasses
from enum import StrEnum
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Any
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
import numpy as np
import pyccl

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.ccl_factory import CAMBExtraParams, PoweSpecAmplitudeParameter


class Frameworks(StrEnum):
    """Statistical analysis frameworks supported by Firecrown.

    Each framework has its own configuration generator that produces
    framework-specific files for cosmological parameter estimation.
    """

    COBAYA = "cobaya"
    COSMOSIS = "cosmosis"
    NUMCOSMO = "numcosmo"


class FrameworkCosmology(StrEnum):
    """Level of cosmology computation required by the framework.

    Controls whether frameworks include Boltzmann solvers (CAMB/CLASS)
    and what level of computation they perform.
    """

    NONE = "none"
    BACKGROUND = "background"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class PriorUniform(BaseModel):
    """Uniform (flat) prior over a bounded interval [lower, upper]."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float

    def model_post_init(self, _, /) -> None:
        """Validate that lower < upper."""
        if not self.lower < self.upper:
            raise ValueError("lower must be < upper")


class PriorGaussian(BaseModel):
    """Gaussian (normal) prior with specified mean and standard deviation."""

    model_config = ConfigDict(extra="forbid")

    mean: float
    sigma: float

    def model_post_init(self, _, /) -> None:
        """Validate that sigma > 0."""
        if not self.sigma > 0:
            raise ValueError("sigma must be > 0")


Prior = PriorUniform | PriorGaussian


class Parameter(BaseModel):
    """Model parameter with sampling metadata.

    Defines a single parameter for cosmological or systematic modeling,
    including its prior bounds, default value, and whether it's free or fixed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    symbol: str
    lower_bound: float
    upper_bound: float
    scale: float = 0.0
    abstol: float = 0.0
    default_value: float
    free: bool
    prior: Prior | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_defaults(cls, data: Any) -> Any:
        """Validate that scale is positive if provided."""
        sd = 0.01  # Default scale factor
        if isinstance(data, dict):
            if (
                ("scale" not in data)
                and ("default_value" in data)
                and data["default_value"] != 0.0
            ):
                data["scale"] = np.abs(data["default_value"]) * sd
            if (
                ("scale" not in data)
                and ("lower_bound" in data)
                and ("upper_bound" in data)
            ):
                data["scale"] = np.abs(data["upper_bound"] - data["lower_bound"]) * sd
        return data

    @model_validator(mode="after")
    def validate_bounds(self) -> "Parameter":
        """Validate that lower_bound < upper_bound."""
        if not self.lower_bound < self.upper_bound:
            raise ValueError("lower_bound must be < upper_bound")
        return self

    @classmethod
    def from_tuple(
        cls,
        name: str,
        symbol: str,
        lower_bound: float,
        upper_bound: float,
        default_value: float,
        free: bool,
        prior: Prior | None = None,
    ):
        """Create Parameter from tuple of values."""
        return cls(
            name=name,
            symbol=symbol,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            default_value=default_value,
            free=free,
            prior=prior,
        )


class Model(BaseModel):
    """Model definition with multiple parameters.

    Groups related parameters (e.g., all photo-z shifts, all galaxy biases)
    into a named model for organization in configuration files.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: list[Parameter]

    _param_dict: dict[str, Parameter] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context, /) -> None:
        """Validate that parameter names are unique.

        Creates a dictionary of parameters by name.
        """
        super().model_post_init(context)

        for param in self.parameters:
            if param.name in self._param_dict:
                raise ValueError(f"Duplicate parameter name: {param.name}")
            self._param_dict[param.name] = param

    def has_priors(self) -> bool:
        """Check if any parameters have priors defined.

        :return: True if at least one parameter has a prior; False otherwise
        """
        for param in self.parameters:
            if param.prior is not None:
                return True
        return False

    def __getitem__(self, key: str) -> Parameter:
        """Return the value for the given key.

        If the key has not been used, add it to the set of used keys.

        :param key: key
        :return: value
        """
        if key not in self._param_dict:
            raise KeyError(f"Parameter {key} not found in model {self.name}")
        return self._param_dict[key]

    def __contains__(self, key: str) -> bool:
        """Return True if the key is in the map, False otherwise.

        :param key: key
        :return: True if the key is in the map, False otherwise
        """
        return key in self._param_dict


CCL_COSMOLOGY_MINIMAL_SET = [
    "Omega_c",
    "Omega_b",
    "Omega_k",
    "h",
    "n_s",
    "Neff",
    "m_nu",
    "w0",
    "wa",
]
CCL_COSMOLOGY_PARAMETERS = CCL_COSMOLOGY_MINIMAL_SET + [
    "sigma8",
    "T_CMB",
    "A_s",
    "T_ncdm",
]

COSMO_DESC = {
    "Omega_c": Parameter.from_tuple("Omega_c", r"\Omega_c", 0.06, 0.46, 0.25, True),
    "Omega_b": Parameter.from_tuple("Omega_b", r"\Omega_b", 0.03, 0.07, 0.05, True),
    "h": Parameter.from_tuple("h", r"h", 0.55, 0.85, 0.67, False),
    "n_s": Parameter.from_tuple("n_s", r"n_s", 0.87, 1.07, 0.96, False),
    "sigma8": Parameter.from_tuple("sigma8", r"\sigma_8", 0.6, 1.0, 0.81, True),
    "A_s": Parameter.from_tuple("A_s", r"A_s", 1e-10, 3e-9, 2.1e-9, True),
    "Omega_k": Parameter.from_tuple("Omega_k", r"\Omega_k", -0.2, 0.2, 0.0, False),
    "Neff": Parameter.from_tuple("Neff", r"N_\mathrm{eff}", 2.0, 5.0, 3.046, False),
    "m_nu": Parameter.from_tuple("m_nu", r"m_\nu", 0.0, 5.0, 0.0, False),
    "w0": Parameter.from_tuple("w0", r"w_0", -3.0, 0.0, -1.0, False),
    "wa": Parameter.from_tuple("wa", r"w_a", -1.0, 1.0, 0.0, False),
    "T_CMB": Parameter.from_tuple("T_CMB", r"T_\mathrm{CMB}", 2.0, 3.0, 2.7255, False),
    "T_ncdm": Parameter.from_tuple(
        "T_ncdm", r"T_\mathrm{ncdm}", 0.5, 1.0, 0.71611, False
    ),
}


class CCLCosmologySpec(Model):
    """CCL cosmology specification model.

    A specialized Model that represents the cosmological parameters
    used in CCL-based analyses.
    """

    name: str = "ccl_cosmology"
    description: str = "CCL cosmology specification"

    # CCL cosmological parameters
    mass_split: str = "normal"
    transfer_function: str | Any = "boltzmann_camb"
    matter_power_spectrum: str | Any = "halofit"
    extra_parameters: CAMBExtraParams | None = None

    def model_post_init(self, context, /) -> None:
        """Validate that all parameters are cosmological."""
        super().model_post_init(context)
        for param in self.parameters:
            if param.name not in CCL_COSMOLOGY_PARAMETERS:
                raise ValueError(
                    f"Parameter '{param.name}' is not a "
                    f"valid CCL cosmological parameter."
                )

        # Has at least the minimal set of parameters
        for req_param in CCL_COSMOLOGY_MINIMAL_SET:
            if req_param not in self._param_dict:
                raise ValueError(
                    f"CCLCosmologySpec is missing required parameter '{req_param}'."
                )
        if ("sigma8" in self._param_dict) == ("A_s" in self._param_dict):
            raise ValueError("Exactly one of A_s and sigma8 must be supplied.")

    def to_ccl_cosmology(self) -> pyccl.Cosmology:
        """Convert to CCL cosmology dictionary.

        :return: Dictionary of cosmological parameters for CCL
        """
        args: dict[str, Any] = {}
        if self._param_dict is not None:
            args.update(
                {key: value.default_value for key, value in self._param_dict.items()}
            )
        if self.extra_parameters:
            args["extra_parameters"] = {"camb": self.extra_parameters.model_dump()}
        if self.matter_power_spectrum:
            args["matter_power_spectrum"] = self.matter_power_spectrum
        if self.transfer_function:
            args["transfer_function"] = self.transfer_function
        if self.mass_split:
            args["mass_split"] = self.mass_split

        return pyccl.Cosmology(**args)

    @classmethod
    def vanilla_lcdm(cls) -> "CCLCosmologySpec":
        """Create a vanilla LCDM cosmology analysis spec with standard parameters."""
        vanilla_params = [
            COSMO_DESC[param_name] for param_name in CCL_COSMOLOGY_MINIMAL_SET
        ] + [COSMO_DESC["sigma8"]]
        return cls(parameters=vanilla_params)

    @classmethod
    def vanilla_lcdm_with_neutrinos(cls) -> "CCLCosmologySpec":
        """Create a vanilla LCDM cosmology analysis spec with standard parameters."""
        parameters = (
            [
                COSMO_DESC[param_name]
                for param_name in CCL_COSMOLOGY_MINIMAL_SET
                if param_name != "m_nu"
            ]
            + [
                COSMO_DESC["m_nu"].model_copy(
                    update={"default_value": 0.06, "free": True}
                )
            ]
            + [COSMO_DESC["sigma8"]]
        )

        return cls(parameters=parameters)

    def get_num_massive_neutrinos(self) -> int:
        """Get the number of massive neutrinos defined in the cosmology."""
        mnu_param = self._param_dict["m_nu"]
        return 1 if (mnu_param.default_value > 0.0 or mnu_param.free) else 0

    def get_amplitude_parameter(self) -> PoweSpecAmplitudeParameter:
        """Get the amplitude parameter for the power spectrum."""
        if "A_s" in self._param_dict:
            return PoweSpecAmplitudeParameter.AS
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
    cosmo_spec: CCLCosmologySpec
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
        """Add likelihood factory source to generator state.

        :param factory_source: Path to factory Python file or YAML module string
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
    """Convert Path to string representation.

    :param path: Path object or string
    :param use_absolute: If True, return absolute path; otherwise return filename only
    :return: Path string suitable for configuration files
    """
    if isinstance(path, str):
        return path
    return path.absolute().as_posix() if use_absolute else path.name
