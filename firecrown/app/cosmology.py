"""Cosmology configuration generator.

This module provides a CLI command to generate cosmology configuration files for
parameter estimation and inference. The generated YAML contains cosmological
parameter specifications with default values and optional prior constraints.

**Supported Prior Types:**

The configuration supports two types of priors that constrain parameter values
during inference:

1. **Gaussian Prior**: Constrains parameter around a mean with sigma (standard
   deviation) ```yaml parameter_name:
     default_value: 0.06
     prior:
       mean: 0.06
       sigma: 0.01
   ```

2. **Uniform Prior**: Constrains parameter between lower and upper bounds
   ```yaml
   parameter_name:
     default_value: 0.25
     prior:
       lower: 0.2
       upper: 0.3
   ```

**Editing Generated Configurations:**

The generated YAML can be manually edited to refine parameter values and priors.
You may:
- Modify ``default_value`` to change initial parameter values
- Adjust ``mean`` and ``sigma`` for Gaussian priors
- Adjust ``lower`` and ``upper`` bounds for Uniform priors
- Add new priors to parameters that initially have none
- Change prior type (Gaussian ↔ Uniform) by modifying the ``prior`` section

**Example workflow:**

1. Generate base configuration: ``firecrown cosmology --cosmology vanilla_lcdm
   config.yaml``
2. Edit ``config.yaml`` to add priors or adjust parameter values
3. Use configuration in parameter estimation pipeline
"""

from typing import Annotated, assert_never
import dataclasses
from pathlib import Path
from enum import StrEnum
import typer
from rich.syntax import Syntax
from rich.panel import Panel
from pydantic import BaseModel, ConfigDict
import yaml
import pyccl

from firecrown.ccl_factory import CAMBExtraParams
from . import logging
from .analysis import CCLCosmologySpec, Prior


class Cosmology(StrEnum):
    """Supported cosmologies."""

    VANILLA_LCDM = "vanilla_lcdm"
    VANILLA_LCDM_WITH_NEUTRINOS = "vanilla_lcdm_with_neutrinos"


class PriorWrapper(BaseModel):
    """Wrapper for prior dictionary."""

    model_config = ConfigDict(extra="forbid")
    value: Prior


def _parse_key_value(key_value_str: str) -> tuple[str, float | None]:
    """Parse key=value string into key and optional value.

    :param key_value_str: String in format 'key' or 'key=value'
    :return: Tuple of (key, value) where value is None if not specified
    :raises ValueError: If value cannot be converted to float
    """
    if "=" not in key_value_str:
        return key_value_str, None

    key, value_str = key_value_str.split("=", 1)
    try:
        value = float(value_str)
    except ValueError as e:
        raise ValueError(
            f"Invalid value '{value_str}' for key '{key}': must be a number"
        ) from e
    return key, value


def _parse_prior_dict(prior_str: str) -> Prior:
    """Parse prior specification string into Prior object.

    Parses comma-separated key=value pairs and validates with PriorWrapper.

    Example:
        'mean=0.06,sigma=0.01' → Gaussian prior with mean=0.06, sigma=0.01
        'lower=0.01,upper=0.1' → Uniform prior with lower=0.01, upper=0.1

    :param prior_str: Prior specification as comma-separated key=value pairs
    :return: Validated Prior object
    :raises ValueError: If prior_str is empty or invalid format
    """
    if not prior_str:
        raise ValueError("Prior specification cannot be empty")

    prior_dict = {}
    for element in prior_str.split(","):
        if "=" not in element:
            raise ValueError(
                f"Invalid prior element '{element}': must be in format 'key=value'"
            )
        k, v = element.split("=", 1)
        try:
            prior_dict[k] = float(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid value '{v}' for prior '{k}': must be a number"
            ) from e

    return PriorWrapper.model_validate({"value": prior_dict}).value


def _parse_prior(prior_arg: str) -> tuple[str, float | None, Prior | None]:
    """Parse cosmology parameter prior specification from command-line argument.

    Parses arguments in the format:
    - 'key=value' → Fixed parameter value, no prior
    - 'key=value,mean=...,sigma=...' → Value with Gaussian prior
    - 'key=value,lower=...,upper=...' → Value with uniform prior
    - 'key,mean=...,sigma=...' → Prior-only constraint (no fixed value)
    - 'key,lower=...,upper=...' → Prior-only constraint (no fixed value)

    Examples:
        'm_nu=0.06,mean=0.06,sigma=0.01'
        'Omega_c=0.26,lower=0.2,upper=0.3'
        'sigma8,mean=0.8,sigma=0.1'

    :param prior_arg: Command-line prior specification string
    :return: Tuple of (parameter_name, default_value, prior_constraint)
        where default_value and prior_constraint may be None
    :raises ValueError: If format is invalid or required fields missing
    """
    # Split parameter/value from prior specification
    if "," in prior_arg:
        key_value_part, prior_part = prior_arg.split(",", 1)
    else:
        key_value_part = prior_arg
        prior_part = None

    # Parse key and optional value
    key, value = _parse_key_value(key_value_part)

    # Parse prior if specified
    if prior_part is None:
        prior = None
    else:
        prior = _parse_prior_dict(prior_part)

    # Validate that we have at least a value or a prior
    if value is None and prior is None:
        raise ValueError(
            f"Parameter '{key}' must have either a default value (key=value) "
            "or prior constraint (key,mean=...)"
        )

    return key, value, prior


@dataclasses.dataclass(kw_only=True)
class Generate(logging.Logging):
    """Cosmology configuration generator."""

    output_file: Annotated[
        Path,
        typer.Argument(
            help="Output file path for generated configuration.",
        ),
    ]

    cosmology: Annotated[
        Cosmology,
        typer.Option(
            "--cosmology",
            "-c",
            help="Cosmology to generate configuration for.",
        ),
    ]

    camb_halofit: Annotated[
        str | None,
        typer.Option(
            "--camb-halofit",
            help="Add CAMB halofit extra parameters to the cosmology.",
        ),
    ] = None

    parameter: Annotated[
        list[str],
        typer.Option(
            "--parameter",
            "-p",
            help=(
                "Update parameter values and/or priors. Can be used to set default "
                "values, add prior constraints, or both. Use multiple times. "
                "Examples: --parameter m_nu=0.06,mean=0.06,sigma=0.01 "
                "--parameter Omega_c=0.26,lower=0.2,upper=0.3 "
                "--parameter sigma8,mean=0.8,sigma=0.1"
            ),
            default_factory=list,
        ),
    ]

    exclude_defaults: Annotated[
        bool,
        typer.Option(
            "--exclude-defaults",
            "-e",
            help="Exclude fields with default values.",
        ),
    ] = False

    print_output: Annotated[
        bool,
        typer.Option(
            "--print-output",
            help="Print generated YAML to console (in addition to file).",
        ),
    ] = False

    def __post_init__(self):
        """Initialize and execute the complete cosmology generation workflow."""
        super().__post_init__()
        match self.cosmology:
            case Cosmology.VANILLA_LCDM:
                spec = CCLCosmologySpec.vanilla_lcdm()
            case Cosmology.VANILLA_LCDM_WITH_NEUTRINOS:
                spec = CCLCosmologySpec.vanilla_lcdm_with_neutrinos()
            case _ as unreachable:
                assert_never(unreachable)

        if self.camb_halofit:
            match self.camb_halofit.lower():
                case "mead":
                    spec.extra_parameters = CAMBExtraParams(
                        HMCode_A_baryon=3.13,
                        HMCode_eta_baryon=0.603,
                        dark_energy_model="ppf",
                        halofit_version="mead",
                        kmax=10.0,
                        lmax=0,
                    )
                case "mead2020_feedback":
                    spec.extra_parameters = CAMBExtraParams(
                        HMCode_logT_AGN=7.8,
                        dark_energy_model="ppf",
                        halofit_version="mead2020_feedback",
                        kmax=10.0,
                        lmax=0,
                    )
                case _:
                    spec.extra_parameters = CAMBExtraParams(
                        dark_energy_model="ppf",
                        halofit_version=self.camb_halofit,
                        kmax=10.0,
                        lmax=0,
                    )

        for param_spec in self.parameter:
            key, value, prior_obj = _parse_prior(param_spec)
            if key not in spec:
                raise ValueError(
                    f"Unknown parameter {key} for cosmology {self.cosmology}"
                )
            if value is not None:
                spec[key].default_value = value
            if prior_obj is not None:
                spec[key].prior = prior_obj

        spec_dump = spec.model_dump(exclude_defaults=self.exclude_defaults)
        yaml_str = yaml.dump(spec_dump, sort_keys=False, width=88)
        self.output_file.write_text(yaml_str)

        ccl_cosmo = spec.to_ccl_cosmology()
        assert ccl_cosmo is not None
        assert isinstance(ccl_cosmo, pyccl.Cosmology)

        self.console.print(f"Configuration written to {self.output_file}")

        if self.print_output:
            yaml_syntax = Syntax(yaml_str, "yaml", theme="ansi_light", word_wrap=True)
            self.console.print(yaml_syntax)

        # Print guidance for editing the configuration
        instructions = (
            "To add or modify priors, edit the YAML file and add 'prior' sections "
            "to parameters. Examples:\n\n"
            "[bold]Gaussian Prior:[/bold] Add to parameter:\n"
            "  prior:\n"
            "    kind: gaussian\n"
            "    mean: 0.06\n"
            "    sigma: 0.01\n\n"
            "[bold]Uniform Prior:[/bold] Add to parameter:\n"
            "  prior:\n"
            "    kind: uniform\n"
            "    lower: 0.2\n"
            "    upper: 0.3"
        )
        self.console.print(Panel(instructions, title="Editing Priors", expand=False))
