"""Helper functions for Cobaya YAML file generation.

This module provides utilities for generating Cobaya configuration files
for cosmological parameter estimation with Firecrown likelihoods.
"""

from typing import Dict, Any
from pathlib import Path

import yaml

import firecrown.connector.cobaya.likelihood
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Model


def create_standard_cobaya_config(
    factory_path: Path,
    build_parameters: NamedParameters,
    likelihood_name: str,
    use_absolute_path: bool = False,
) -> Dict[str, Any]:
    """Create standard Cobaya configuration for cosmic shear analysis.

    :param factory_path: Path to the factory file
    :param sacc_path: Path to the SACC data file
    :param likelihood_name: Name for the likelihood in configuration
    :param use_absolute_path: Whether to use absolute paths
    :return: Dictionary containing Cobaya configuration
    """

    if use_absolute_path:
        factory_filename = factory_path.absolute().as_posix()
    else:
        factory_filename = factory_path.name

    config = {
        "theory": {
            "camb": {
                "stop_at_error": True,
                "extra_args": {"num_massive_neutrinos": 1, "halofit_version": "mead"},
            }
        },
        "likelihood": {
            likelihood_name: {
                "input_style": "CAMB",
                "external": firecrown.connector.cobaya.likelihood.LikelihoodConnector,
                "firecrownIni": factory_filename,
                "build_parameters": build_parameters.convert_to_basic_dict(),
            }
        },
        "params": _get_standard_params(),
        "sampler": {"evaluate": None},
        "stop_at_error": True,
        "output": "output",
        "packages_path": None,
        "test": False,
        "debug": False,
    }

    return config


def _get_standard_params() -> Dict[str, Any]:
    """Generate standard parameter configuration for cosmic shear.

    :param n_bins: Number of tomographic bins
    :return: Dictionary of parameter configurations
    """
    params = {
        # Cosmological parameters
        "sigma8": {"prior": {"min": 0.7, "max": 1.2}, "ref": 0.801, "proposal": 0.801},
        "ombh2": 0.01860496,
        "omch2": {
            "prior": {"min": 0.05, "max": 0.2},
            "ref": 0.120932240,
            "proposal": 0.01,
        },
        "omk": 0.0,
        "TCMB": 2.7255,
        "H0": 68.2,
        "mnu": 0.06,
        "nnu": 3.046,
        "ns": 0.971,
        "w": -1.0,
        "wa": 0.0,
    }

    return params


def add_models_to_cobaya_config(config: Dict[str, Any], models: list[Model]) -> None:
    """Add model parameters to Cobaya configuration.

    :param config: Configuration dictionary to modify
    :param models: List of models with parameters
    """
    for model in models:
        for parameter in model.parameters:
            if parameter.free:
                config["params"][parameter.name] = {
                    "ref": parameter.default_value,
                    "prior": {
                        "min": parameter.lower_bound,
                        "max": parameter.upper_bound,
                    },
                }
            else:
                config["params"][parameter.name] = parameter.default_value


def write_cobaya_config(config: Dict[str, Any], output_file: Path) -> None:
    """Write Cobaya configuration to YAML file.

    :param config: Configuration dictionary
    :param output_file: Output YAML file path
    """
    with output_file.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
