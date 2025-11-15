"""CosmoSIS configuration file generator.

Generates CosmoSIS INI files for cosmological parameter estimation with Firecrown.
Produces pipeline configuration, parameter values, and optional priors files.

CosmoSIS uses INI format with extended interpolation and requires specific
formatting (e.g., floats with .0 suffix). Gaussian priors in values files are
converted to uniform bounds (mean ± 3σ), with true Gaussian priors in separate file.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

from typing import assert_never
import configparser
import textwrap
from pathlib import Path
import dataclasses

import numpy as np

import firecrown
from firecrown.likelihood.likelihood import NamedParameters

from ._types import (
    Model,
    Frameworks,
    ConfigGenerator,
    FrameworkCosmology,
    CCLCosmologyAnalysisSpec,
    Prior,
    PriorGaussian,
    PriorUniform,
    get_path_str,
)

# Section name for cosmological parameters in CosmoSIS INI files
COSMOLOGICAL_PARAMETERS = "cosmological_parameters"

# Map CCL parameter names to CosmoSIS parameter names
NAME_MAP = {
    "Omega_c": "omega_c",
    "Omega_b": "omega_b",
    "Omega_k": "omega_k",
    "h": "h0",
    "w0": "w",
    "wa": "wa",
    "sigma8": "sigma_8",
    "A_s": "A_s",
    "n_s": "n_s",
}


def format_comment(text: str, width: int = 88) -> list[str]:
    """Format text as CosmoSIS comment lines with ;; prefix.

    Wraps long text into multiple lines, each prefixed with ';; '.

    :param text: Comment text to format
    :param width: Maximum line width including prefix
    :return: List of formatted comment lines
    """
    # Account for ";; " prefix (3 characters)
    wrap_width = width - 3
    wrapped_lines = textwrap.wrap(text, width=wrap_width)
    return [f";; {line}" for line in wrapped_lines]


def add_comment_block(
    config: configparser.ConfigParser, section: str, text: str
) -> None:
    """Add formatted comment block to configuration section.

    Comments are added as keys with no values, which ConfigParser
    preserves when writing to file.

    :param config: ConfigParser object (modified in-place)
    :param section: Section name where comment will be added
    :param text: Comment text to format and add
    """
    for comment_line in format_comment(text):
        config.set(section, comment_line)


def _add_cosmology_modules(
    cfg: configparser.ConfigParser,
    required_cosmology: FrameworkCosmology,
    cosmo_spec: CCLCosmologyAnalysisSpec,
) -> None:
    """Add CAMB and consistency modules to pipeline configuration."""
    cfg["consistency"] = {
        "file": "${CSL_DIR}/utility/consistency/consistency_interface.py",
    }
    if required_cosmology == FrameworkCosmology.BACKGROUND:
        cfg["camb"] = {
            "file": "${CSL_DIR}/boltzmann/camb/camb_interface.py",
            "mode": "background",
            "feedback": "0",
        }
    else:
        cfg["camb"] = {
            "file": "${CSL_DIR}/boltzmann/camb/camb_interface.py",
            "mode": "power",
            "feedback": "0",
            "zmin": "0.0",
            "zmax": "4.0",
            "nz": "100",
            "kmax": "50.0",
            "nk": "1000",
            **(
                cosmo_spec.cosmology.extra_parameters.get_dict()
                if cosmo_spec.cosmology.extra_parameters
                else {}
            ),
        }


def _add_firecrown_likelihood(
    cfg: configparser.ConfigParser,
    factory_source_str: str,
    build_parameters: NamedParameters,
) -> None:
    """Add Firecrown likelihood module to pipeline configuration."""
    cfg["firecrown_likelihood"] = {}
    add_comment_block(
        cfg,
        "firecrown_likelihood",
        "This will point to the Firecrown's cosmosis likelihood connector module.",
    )
    firecrown_path = Path(firecrown.__path__[0])
    cfg.set(
        "firecrown_likelihood",
        "file",
        f"{firecrown_path}/connector/cosmosis/likelihood.py",
    )
    cfg.set("firecrown_likelihood", "likelihood_source", factory_source_str)
    for key, value in build_parameters.convert_to_basic_dict().items():
        cfg.set("firecrown_likelihood", key, str(value))


def create_config(
    prefix: str,
    factory_source: str | Path,
    build_parameters: NamedParameters,
    values_path: Path,
    priors_path: Path | None,
    output_path: Path,
    cosmo_spec: CCLCosmologyAnalysisSpec,
    *,
    use_absolute_path: bool = True,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
) -> configparser.ConfigParser:
    """Create CosmoSIS pipeline configuration (main INI file).

    Configures runtime, pipeline modules, sampler, and likelihood connector.
    Optionally includes CAMB and consistency modules for cosmology computation.

    :param prefix: Filename prefix for output files
    :param factory_source: Path to factory file or YAML module string
    :param build_parameters: Parameters passed to likelihood factory
    :param values_path: Path to values.ini file (referenced in pipeline)
    :param priors_path: Path to priors.ini file (optional)
    :param output_path: Output directory for results
    :param cosmo_spec: Cosmology specification
    :param use_absolute_path: Use absolute paths in configuration
    :param required_cosmology: Level of cosmology computation
    :return: Configured ConfigParser ready to write
    """
    cfg = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(), allow_no_value=True
    )

    use_cosmology = required_cosmology != FrameworkCosmology.NONE

    cfg["runtime"] = {
        "sampler": "test",
        "root": str(output_path.absolute()),
        "verbosity": "quiet",
    }
    cfg["DEFAULT"] = {"fatal_errors": "T"}
    cfg["output"] = {
        "filename": f"output/{prefix}_samples.txt",
        "format": "text",
        "verbosity": "0",
    }
    cfg["pipeline"] = {
        "modules": (
            "consistency camb firecrown_likelihood"
            if use_cosmology
            else "firecrown_likelihood"
        ),
        "values": get_path_str(values_path, use_absolute_path),
        "likelihoods": "firecrown",
        "debug": "T",
        "timing": "T",
    }
    if priors_path is not None:
        cfg["pipeline"]["priors"] = get_path_str(priors_path, use_absolute_path)

    if use_cosmology:
        _add_cosmology_modules(cfg, required_cosmology, cosmo_spec)

    _add_firecrown_likelihood(
        cfg, get_path_str(factory_source, use_absolute_path), build_parameters
    )

    cfg["test"] = {"fatal_errors": "T", "save_dir": "output"}

    return cfg


def add_models(config: configparser.ConfigParser, models: list[Model]) -> None:
    """Register model parameter sections in pipeline configuration.

    Adds sampling_parameters_sections to firecrown_likelihood module,
    telling CosmoSIS which sections contain parameters to sample.

    :param config: Pipeline configuration (modified in-place)
    :param models: List of models with parameters
    """
    model_list = [model.name for model in models]
    config.set(
        "firecrown_likelihood",
        "sampling_parameters_sections",
        " ".join(model_list),
    )


def format_float(value: float) -> str:
    """Format float for CosmoSIS compatibility.

    Formats to 3 significant digits and ensures decimal point is present
    (adds '.0' if needed). CosmoSIS requires floats to have decimal points.

    :param value: Numeric value to format
    :return: Formatted string (e.g., '0.67', '1.0', '2.5e-9')
    """
    val = f"{value:.3g}"
    if "." not in val:
        val += ".0"
    return val


def _configure_parameter(
    default_value: float,
    prior: PriorGaussian | PriorUniform | None,
    scale: float = 1.0,
) -> str:
    """Format parameter for CosmoSIS values file.

    Returns either a fixed value or 'min start max' for sampled parameters.
    Gaussian priors are converted to uniform bounds (mean ± 3σ) since
    CosmoSIS values files don't support Gaussian priors directly.

    :param default_value: Default/reference parameter value
    :param prior: Prior specification (None for fixed parameters)
    :param scale: Scale factor applied to all values
    :return: Formatted parameter string
    """
    if prior is None:
        return format_float(default_value * scale)

    match prior:
        case PriorGaussian():
            # CosmoSIS doesn't support Gaussian priors in values file
            # Use mean ± 3*sigma as uniform bounds
            lower = (prior.mean - 3 * prior.sigma) * scale
            upper = (prior.mean + 3 * prior.sigma) * scale
            ref = default_value * scale
            return f"{format_float(lower)} {format_float(ref)} {format_float(upper)}"
        case PriorUniform():
            ref = default_value * scale
            return (
                f"{format_float(prior.lower * scale)} "
                f"{format_float(ref)} "
                f"{format_float(prior.upper * scale)}"
            )
        case _ as unreachable:
            assert_never(unreachable)


def add_cosmological_parameters(
    config: configparser.ConfigParser,
    cosmo_spec: CCLCosmologyAnalysisSpec,
) -> None:
    """Add cosmological parameters to values configuration.

    Creates 'cosmological_parameters' section with parameter values and bounds.
    Includes fixed tau=0.08 required by CosmoSIS CAMB module.

    :param config: Values ConfigParser (modified in-place)
    :param cosmo_spec: Cosmology specification with parameters and priors
    """
    section = COSMOLOGICAL_PARAMETERS
    config.add_section(section)

    add_comment_block(
        config,
        section,
        "Parameters and data in CosmoSIS are organized into sections "
        "so we can easily see what they mean. "
        "This file contains cosmological parameters "
        "and firecrown-specific parameters.",
    )

    cosmo = cosmo_spec.cosmology.to_ccl_cosmology()
    priors = cosmo_spec.priors

    name_map: dict[str, tuple[float, Prior | None]] = {
        "Omega_c": (1.0, priors.Omega_c),
        "Omega_b": (1.0, priors.Omega_b),
        "Omega_k": (1.0, priors.Omega_k),
        "h": (1.0, priors.h),
        "w0": (1.0, priors.w0),
        "wa": (1.0, priors.wa),
        "sigma8": (1.0, priors.sigma8),
        "A_s": (1.0, priors.A_s),
        "n_s": (1.0, priors.n_s),
    }

    for param, (scale, prior) in name_map.items():
        if cosmo[param] is None or np.isnan(cosmo[param]):
            continue
        name = NAME_MAP[param]
        config.set(section, name, _configure_parameter(cosmo[param], prior, scale))
    # Fixed value for reionization optical depth, CCL does not support this but
    # CosmoSIS CAMB module requires it.
    config.set(section, "tau", "0.08")


def add_firecrown_models(
    config: configparser.ConfigParser, models: list[Model]
) -> None:
    """Add systematic/nuisance model parameters to values configuration.

    Creates a section for each model with its parameters formatted as
    fixed values or 'min start max' for free parameters.

    :param config: Values ConfigParser (modified in-place)
    :param models: List of models with parameters
    """
    for model in models:
        section = model.name
        config.add_section(section)
        add_comment_block(
            config,
            section,
            f"Firecrown parameters for the {model.name} model.",
        )
        for parameter in model.parameters:
            if parameter.free:
                config.set(
                    section,
                    parameter.name,
                    format_float(parameter.lower_bound)
                    + " "
                    + format_float(parameter.default_value)
                    + " "
                    + format_float(parameter.upper_bound),
                )
            else:
                # CosmoSIS does not like integers
                config.set(
                    section, parameter.name, format_float(parameter.default_value)
                )


def create_values_config(
    cosmo_spec: CCLCosmologyAnalysisSpec,
    models: list[Model] | None = None,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
) -> configparser.ConfigParser:
    """Create values.ini with parameter values and sampling bounds.

    Generates configuration with cosmological parameters (if cosmology required)
    and model parameters. Free parameters get 'min start max', fixed get single value.

    :param cosmo_spec: Cosmology specification
    :param models: List of systematic/nuisance models
    :param required_cosmology: Level of cosmology computation
    :return: Configured ConfigParser ready to write
    """
    config = configparser.ConfigParser(allow_no_value=True)

    if required_cosmology != FrameworkCosmology.NONE:
        add_cosmological_parameters(config, cosmo_spec)

    if models:
        add_firecrown_models(config, models)

    return config


def add_cosmological_parameters_priors(
    config: configparser.ConfigParser,
    cosmo_spec: CCLCosmologyAnalysisSpec,
) -> None:
    """Add cosmological parameter priors to priors configuration.

    Formats priors as 'gaussian mean sigma' or 'uniform min max'.
    Only parameters with defined priors are included.

    :param config: Priors ConfigParser (modified in-place)
    :param cosmo_spec: Cosmology specification with priors
    """
    priors = cosmo_spec.priors
    section = COSMOLOGICAL_PARAMETERS
    config.add_section(section)
    add_comment_block(
        config, section, "This section contains priors for cosmological parameters."
    )

    for param, name in NAME_MAP.items():
        prior = priors.__dict__[param]
        if prior is None:
            continue
        match prior:
            case PriorGaussian():
                assert isinstance(prior, PriorGaussian)
                config.set(
                    section,
                    name,
                    f"gaussian {format_float(prior.mean)} "
                    f"{format_float(prior.sigma)}",
                )
            case PriorUniform():
                config.set(
                    section,
                    name,
                    f"uniform {format_float(prior.lower)} "
                    f"{format_float(prior.upper)}",
                )
            case _ as unreachable:
                assert_never(unreachable)


def create_priors_config(
    cosmo_spec: CCLCosmologyAnalysisSpec,
    _models: list[Model] | None = None,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
) -> configparser.ConfigParser | None:
    """Create priors.ini with prior specifications.

    Returns None if no priors are defined (priors file not needed).
    Currently only supports cosmological parameter priors.

    :param cosmo_spec: Cosmology specification with priors
    :param _models: Reserved for future model priors support
    :param required_cosmology: Level of cosmology computation
    :return: Configured ConfigParser or None if no priors
    """
    if cosmo_spec.priors.is_empty():
        return None

    config = configparser.ConfigParser(allow_no_value=True)

    if required_cosmology != FrameworkCosmology.NONE:
        add_cosmological_parameters_priors(config, cosmo_spec)

    return config


@dataclasses.dataclass
class CosmosisConfigGenerator(ConfigGenerator):
    """CosmoSIS configuration generator.

    Generates CosmoSIS INI files for parameter estimation:
    - cosmosis_{prefix}.ini: Pipeline configuration with modules and settings
    - cosmosis_{prefix}_values.ini: Parameter values and sampling bounds
    - cosmosis_{prefix}_priors.ini: Prior specifications (if priors defined)
    """

    framework = Frameworks.COSMOSIS

    def write_config(self) -> None:
        """Write CosmoSIS configuration."""
        assert self.factory_source is not None
        assert self.build_parameters is not None

        cosmosis_ini = self.output_path / f"cosmosis_{self.prefix}.ini"
        values_ini = self.output_path / f"cosmosis_{self.prefix}_values.ini"
        priors_ini = self.output_path / f"cosmosis_{self.prefix}_priors.ini"

        values_cfg = create_values_config(
            self.cosmo_spec, self.models, self.required_cosmology
        )
        priors_cfg = create_priors_config(
            self.cosmo_spec, self.models, self.required_cosmology
        )

        cfg = create_config(
            prefix=self.prefix,
            factory_source=self.factory_source,
            build_parameters=self.build_parameters,
            values_path=values_ini,
            priors_path=priors_ini if priors_cfg is not None else None,
            output_path=self.output_path,
            cosmo_spec=self.cosmo_spec,
            use_absolute_path=self.use_absolute_path,
            required_cosmology=self.required_cosmology,
        )
        add_models(cfg, self.models)

        with values_ini.open("w") as fp:
            values_cfg.write(fp)
        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)
        if priors_cfg is not None:
            with priors_ini.open("w") as fp:
                priors_cfg.write(fp)
