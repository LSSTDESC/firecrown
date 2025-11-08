"""CosmoSIS configuration file generation utilities.

Provides functions to create standard CosmoSIS .ini files with
proper formatting, comments, and parameter sections.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

import configparser
import textwrap
from pathlib import Path
import firecrown
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Model


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


def create_standard_cosmosis_config(
    prefix: str,
    factory_path: Path,
    build_parameters: NamedParameters,
    values_path: Path,
    output_path: Path,
    model_list: list[str],
    use_absolute_path: bool = True,
) -> configparser.ConfigParser:
    """Create standard CosmoSIS pipeline configuration.

    :param prefix: Filename prefix
    :param factory_path: Path to factory file
    :param build_parameters: Likelihood build parameters
    :param values_path: Path to values.ini file
    :param output_path: Output directory
    :param model_list: List of model section names
    :param use_absolute_path: Use absolute paths
    :return: Configured ConfigParser
    """
    cfg = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(), allow_no_value=True
    )
    if use_absolute_path:
        factory_filename = factory_path.absolute().as_posix()
        values_filename = values_path.absolute().as_posix()
    else:
        factory_filename = factory_path.name
        values_filename = values_path.name

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

    # Pipeline configuration
    cfg["pipeline"] = {
        "modules": "consistency camb firecrown_likelihood",
        "values": values_filename,
        "likelihoods": "firecrown",
        "debug": "T",
        "timing": "T",
    }

    # Consistency module
    cfg["consistency"] = {
        "file": "${CSL_DIR}/utility/consistency/consistency_interface.py",
    }

    # CAMB module
    cfg["camb"] = {
        "file": "${CSL_DIR}/boltzmann/camb/camb_interface.py",
        "mode": "all",
        "lmax": "2500",
        "feedback": "0",
        "zmin": "0.0",
        "zmax": "4.00",
        "nz": "100",
        "kmax": "50.0",
        "nk": "1000",
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

    cfg.set("firecrown_likelihood", "likelihood_source", factory_filename)
    cfg.set("firecrown_likelihood", "require_nonlinear_pk", "True")
    cfg.set(
        "firecrown_likelihood",
        "sampling_parameters_sections",
        " ".join(model_list),
    )

    for key, value in build_parameters.convert_to_basic_dict().items():
        cfg.set("firecrown_likelihood", key, str(value))

    # Sampler configurations
    cfg["test"] = {"fatal_errors": "T", "save_dir": "output"}

    return cfg


def create_standard_values_config(
    models: list[Model] | None = None,
) -> configparser.ConfigParser:
    """Create CosmoSIS values.ini with cosmological and model parameters.

    :param models: List of models with parameters
    :return: Configured ConfigParser
    """
    config = configparser.ConfigParser(allow_no_value=True)

    # Cosmological parameters section
    section = "cosmological_parameters"
    config.add_section(section)

    # Add top-level comments
    add_comment_block(
        config,
        section,
        "Parameters and data in CosmoSIS are organized into sections "
        "so we can easily see what they mean. "
        "This file contains cosmological parameters "
        "and firecrown-specific parameters.",
    )

    # Varied parameters (min, start, max)
    config.set(section, "omega_c", "0.06 0.26 0.46")
    config.set(section, "omega_b", "0.03 0.04 0.07")

    # Fixed parameters section
    add_comment_block(
        config, section, "The following parameters are fixed, not varied in sampling."
    )
    config.set(section, "omega_k", "0.0")
    config.set(section, "tau", "0.08")
    config.set(section, "n_s", "0.971")
    config.set(section, "sigma_8", "0.801")
    config.set(section, "h0", "0.682")
    config.set(section, "w", "-1.0")
    config.set(section, "wa", "0.0")

    # Add firecrown parameters
    if models:
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
                        f"{parameter.lower_bound:.3g} "
                        f"{parameter.default_value:.3g} "
                        f"{parameter.upper_bound:.3g}",
                    )
                else:
                    config.set(
                        section,
                        parameter.name,
                        f"{parameter.default_value:.3g}",
                    )
    return config
