"""CosmoSIS configuration file generation utilities.

Provides functions to create standard CosmoSIS .ini files with
proper formatting, comments, and parameter sections.

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

COSMOLOGICAL_PARAMETERS = "cosmological_parameters"
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

    :param text: Comment text
    :param width: Maximum line width
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

    :param config: ConfigParser object
    :param section: Section name
    :param text: Comment text
    """
    for comment_line in format_comment(text):
        config.set(section, comment_line)


def create_config(
    prefix: str,
    factory_source: str | Path,
    build_parameters: NamedParameters,
    values_path: Path,
    priors_path: Path | None,
    output_path: Path,
    cosmo_spec: CCLCosmologyAnalysisSpec,
    use_absolute_path: bool = True,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
) -> configparser.ConfigParser:
    """Create standard CosmoSIS pipeline configuration.

    :param prefix: Filename prefix
    :param factory_path: Path to factory file
    :param build_parameters: Likelihood build parameters
    :param values_path: Path to values.ini file
    :param output_path: Output directory
    :param use_absolute_path: Use absolute paths
    :param use_cosmology: Include CAMB in pipeline
    :return: Configured ConfigParser
    """
    cfg = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(), allow_no_value=True
    )

    factory_source_str = get_path_str(factory_source, use_absolute_path)
    values_filename = get_path_str(values_path, use_absolute_path)

    # Runtime configuration
    cfg["runtime"] = {
        "sampler": "test",
        "root": str(output_path.absolute()),
        "verbosity": "quiet",
    }
    cfg["DEFAULT"] = {"fatal_errors": "T"}

    # Output configuration
    cfg["output"] = {
        "filename": f"output/{prefix}_samples.txt",
        "format": "text",
        "verbosity": "0",
    }

    use_cosmology = required_cosmology != FrameworkCosmology.NONE

    # Pipeline configuration
    if use_cosmology:
        modules = "consistency camb firecrown_likelihood"
    else:
        modules = "firecrown_likelihood"

    cfg["pipeline"] = {
        "modules": modules,
        "values": values_filename,
        "likelihoods": "firecrown",
        "debug": "T",
        "timing": "T",
    }
    if priors_path is not None:
        priors_filename = get_path_str(priors_path, use_absolute_path)
        cfg["pipeline"]["priors"] = priors_filename

    if use_cosmology:
        # Consistency module
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
            # CAMB module
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

    # Firecrown likelihood module
    cfg["firecrown_likelihood"] = {}

    # Add formatted comments for firecrown section
    add_comment_block(
        cfg,
        "firecrown_likelihood",
        "This will point to the Firecrown's cosmosis likelihood connector module.",
    )

    firecrown_path = Path(firecrown.__path__[0])
    assert firecrown_path.exists(), f"Firecrown path not found: {firecrown_path}"

    cfg.set(
        "firecrown_likelihood",
        "file",
        f"{firecrown_path}/connector/cosmosis/likelihood.py",
    )

    cfg.set("firecrown_likelihood", "likelihood_source", factory_source_str)
    for key, value in build_parameters.convert_to_basic_dict().items():
        cfg.set("firecrown_likelihood", key, str(value))

    # Sampler configurations
    cfg["test"] = {"fatal_errors": "T", "save_dir": "output"}

    return cfg


def add_models(config: configparser.ConfigParser, models: list[Model]) -> None:
    """Add model parameters to CosmoSIS configuration.

    :param config: Configuration object (modified in-place)
    :param models: List of models with parameters
    """
    model_list = [model.name for model in models]
    config.set(
        "firecrown_likelihood",
        "sampling_parameters_sections",
        " ".join(model_list),
    )


def format_float(value: float) -> str:
    """Format a float to 3 significant digits.

    Add a ".0" if no decimal. This is necessary to work with CosmoSIS.

    :param value: Value to format
    :return: Formatted value
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
    """Configure a parameter in a CosmoSIS values file.

    :param default_value: Default parameter value
    :param prior: Prior specification
    :param scale: Scale factor to apply to prior bounds
    :return: CosmoSIS parameter string (fixed or min start max)
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
    """Add cosmological parameters section to CosmoSIS values config.

    :param config: ConfigParser object (modified in-place)
    :param cosmo_spec: Cosmology analysis specification
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
    """Add firecrown model parameters to CosmoSIS values config.

    :param config: ConfigParser object (modified in-place)
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
    """Create CosmoSIS values.ini with cosmological and model parameters.

    :param models: List of models with parameters
    :param amplitude_parameter: Power spectrum amplitude parameter
    :param required_cosmology: Cosmology requirement level
    :return: Configured ConfigParser
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
    """Add cosmological parameters priors to CosmoSIS priors config.

    :param config: ConfigParser object (modified in-place)
    :param cosmo_spec: Cosmology analysis specification
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
    """Create CosmoSIS priors.ini with cosmological and model priors.

    :param models: List of models with parameters
    :param amplitude_parameter: Power spectrum amplitude parameter
    :param required_cosmology: Cosmology requirement level
    :return: Configured ConfigParser
    """
    if cosmo_spec.priors.is_empty():
        return None

    config = configparser.ConfigParser(allow_no_value=True)

    if required_cosmology != FrameworkCosmology.NONE:
        add_cosmological_parameters_priors(config, cosmo_spec)

    return config


@dataclasses.dataclass
class CosmosisConfigGenerator(ConfigGenerator):
    """Generates CosmoSIS .ini configuration files.

    Creates two files:
    - cosmosis_{prefix}.ini: Pipeline configuration
    - cosmosis_{prefix}_values.ini: Parameter values and priors
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
