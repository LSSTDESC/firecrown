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

import firecrown
from firecrown.likelihood.likelihood import NamedParameters

from ._types import (
    Model,
    Frameworks,
    ConfigGenerator,
    FrameworkCosmology,
    CCLCosmologySpec,
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
    "Neff": "Neff",
    "m_nu": "m_nu",
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
    cosmo_spec: CCLCosmologySpec,
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
                cosmo_spec.extra_parameters.get_dict()
                if cosmo_spec.extra_parameters
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
    cosmo_spec: CCLCosmologySpec,
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
    s = f"{value:.3g}"
    return s if ("." in s or "e" in s) else s + ".0"


def add_firecrown_model(
    config: configparser.ConfigParser,
    model: Model,
    section: str | None = None,
    comment: str | None = None,
    name_map: dict[str, str] | None = None,
    param_scale: dict[str, float] | None = None,
) -> None:
    """Add systematic/nuisance model parameters to values configuration.

    Creates a section for each model with its parameters formatted as
    fixed values or 'min start max' for free parameters.

    :param config: Values ConfigParser (modified in-place)
    :param models: List of models with parameters
    """
    section = model.name if section is None else section
    comment = comment or f"Parameters for the {model.name} model."
    config.add_section(section)
    add_comment_block(config, section, comment)
    param_scale = param_scale or {}
    name_map = name_map or {}
    for parameter in model.parameters:
        name = name_map.get(parameter.name, parameter.name)
        if parameter.free:
            scale = param_scale.get(parameter.name, 1.0)
            config.set(
                section,
                name,
                format_float(parameter.lower_bound * scale)
                + " "
                + format_float(parameter.default_value * scale)
                + " "
                + format_float(parameter.upper_bound * scale),
            )
        else:
            # CosmoSIS does not like integers
            config.set(section, name, format_float(parameter.default_value * 1.0))


def create_values_config(
    cosmo_spec: CCLCosmologySpec,
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
        add_firecrown_model(
            config,
            cosmo_spec,
            section=COSMOLOGICAL_PARAMETERS,
            name_map=NAME_MAP,
            comment=(
                "Parameters and data in CosmoSIS are organized into sections "
                "so we can easily see what they mean. "
                "This file contains cosmological parameters "
                "and firecrown-specific parameters."
            ),
        )
        # Fixed value for reionization optical depth, CCL does not support this but
        # CosmoSIS CAMB module requires it.
        config.set(COSMOLOGICAL_PARAMETERS, "tau", "0.08")
        config.set(
            COSMOLOGICAL_PARAMETERS,
            "num_massive_neutrinos",
            str(cosmo_spec.get_num_massive_neutrinos()),
        )

    if models:
        for model in models:
            add_firecrown_model(config, model)

    return config


def add_model_priors(
    config: configparser.ConfigParser,
    model: Model,
    section: str | None = None,
    comment: str | None = None,
    name_map: dict[str, str] | None = None,
    param_scale: dict[str, float] | None = None,
) -> None:
    """Add cosmological parameter priors to priors configuration.

    Formats priors as 'gaussian mean sigma' or 'uniform min max'.
    Only parameters with defined priors are included.

    :param config: Priors ConfigParser (modified in-place)
    :param cosmo_spec: Cosmology specification with priors
    """
    if not model.has_priors():
        return
    section = model.name if section is None else section
    comment = comment or f"Priors for the {model.name} model."
    config.add_section(section)
    add_comment_block(config, section, comment)

    for param in model.parameters:
        if param.prior is None:
            continue
        name = name_map[param.name] if name_map else param.name
        scale = (
            param_scale[param.name]
            if param_scale and (param.name in param_scale)
            else 1.0
        )
        match param.prior:
            case PriorGaussian():
                config.set(
                    section,
                    param.name,
                    f"gaussian {format_float(param.prior.mean * scale)} "
                    f"{format_float(param.prior.sigma * scale)}",
                )
            case PriorUniform():
                config.set(
                    section,
                    name,
                    f"uniform {format_float(param.prior.lower * scale)} "
                    f"{format_float(param.prior.upper * scale)}",
                )
            case _ as unreachable:
                assert_never(unreachable)


def create_priors_config(
    cosmo_spec: CCLCosmologySpec,
    models: list[Model] | None = None,
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
    if models is None:
        models = []
    if not any(m.has_priors() for m in [cosmo_spec] + models):
        return None

    config = configparser.ConfigParser(allow_no_value=True)

    if (required_cosmology != FrameworkCosmology.NONE) and cosmo_spec.has_priors():
        add_model_priors(
            config, cosmo_spec, section=COSMOLOGICAL_PARAMETERS, name_map=NAME_MAP
        )
    else:
        models = [cosmo_spec] + models

    for model in models:
        add_model_priors(config, model)

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
