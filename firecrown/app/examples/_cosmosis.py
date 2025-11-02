"""Helper functions for CosmoSIS ini file generation.

This module provides utilities for generating CosmoSIS configuration files
with proper comment formatting and standard sections.
"""

import configparser
import textwrap
from typing import Optional
from pathlib import Path
import firecrown


def format_comment(text: str, width: int = 88) -> list[str]:
    """Format a long comment string into wrapped lines with ;; prefix.

    :param text: Comment text to format
    :param width: Maximum line width (default: 88)
    :return: List of formatted comment lines
    """
    # Account for ";; " prefix (3 characters)
    wrap_width = width - 3
    wrapped_lines = textwrap.wrap(text, width=wrap_width)
    return [f";; {line}" for line in wrapped_lines]


def add_comment_block(
    config: configparser.ConfigParser, section: str, text: str
) -> None:
    """Add a formatted comment block to a config section.

    :param config: ConfigParser object
    :param section: Section name to add comments to
    :param text: Comment text to format and add
    """
    for comment_line in format_comment(text):
        config.set(section, comment_line)


def create_standard_cosmosis_config(
    prefix: str,
    factory_filename: str,
    sacc_filename: str,
    values_filename: str,
    output_path: Path,
    n_bins: Optional[int] = None,
) -> configparser.ConfigParser:
    """Create standard CosmoSIS configuration with common sections.

    :param prefix: Prefix for output files
    :param factory_filename: Name of the factory file
    :param sacc_filename: Name of the SACC data file
    :param values_filename: Name of the values ini file
    :param output_path: Path to the output directory
    :param n_bins: Number of tomographic bins (optional)
    :return: Configured ConfigParser object
    """
    cfg = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(), allow_no_value=True
    )

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
        "firecrown_two_point",
    )
    cfg.set("firecrown_likelihood", "sacc_file", sacc_filename)

    if n_bins is not None:
        cfg.set("firecrown_likelihood", "n_bins", str(n_bins))

    # Sampler configurations
    cfg["test"] = {"fatal_errors": "T", "save_dir": "output"}

    return cfg


def create_standard_values_config() -> configparser.ConfigParser:
    """Create standard values.ini configuration for cosmological parameters.

    :param n_bins: Number of tomographic bins for firecrown parameters
    :return: Configured ConfigParser object for values
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

    return config
