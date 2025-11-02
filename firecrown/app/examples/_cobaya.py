"""Helper functions for Cobaya YAML file generation.

This module provides utilities for generating Cobaya configuration files
for cosmological parameter estimation with Firecrown likelihoods.
"""

from typing import Dict, Any
from pathlib import Path

import yaml

import firecrown.connector.cobaya.likelihood


def create_standard_cobaya_config(
    factory_filename: str,
    sacc_filename: str,
    n_bins: int,
    likelihood_name: str = "firecrown_likelihood",
) -> Dict[str, Any]:
    """Create standard Cobaya configuration for cosmic shear analysis.

    :param factory_filename: Name of the factory file
    :param sacc_filename: Name of the SACC data file
    :param n_bins: Number of tomographic bins
    :param likelihood_name: Name for the likelihood in configuration
    :return: Dictionary containing Cobaya configuration
    """
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
                "build_parameters": {"sacc_file": sacc_filename},
            }
        },
        "params": _get_standard_params(n_bins),
        "sampler": {"evaluate": None},
        "stop_at_error": True,
        "output": "output",
        "packages_path": None,
        "test": False,
        "debug": False,
    }

    return config


def _get_standard_params(n_bins: int) -> Dict[str, Any]:
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
        # IA model parameters
        "ia_bias": {"ref": 0.5, "prior": {"min": -5.0, "max": 5.0}},
        "alphaz": {"ref": 0.0, "prior": {"min": -5.0, "max": 5.0}},
        "z_piv": 0.62,
    }

    # Add lens bias parameters
    for i in range(n_bins):
        params[f"lens{i}_bias"] = {
            "ref": 1.4 + i * 0.2,  # Increasing with redshift
            "prior": {"min": 0.8, "max": 3.0},
        }

    # Add photo-z shift parameters for sources
    source_shifts = [-0.001, -0.019, 0.009, -0.018]
    source_scales = [0.016, 0.013, 0.011, 0.022]

    for i in range(min(n_bins, len(source_shifts))):
        params[f"src{i}_delta_z"] = {
            "ref": source_shifts[i],
            "prior": {
                "dist": "norm",
                "loc": source_shifts[i],
                "scale": source_scales[i],
            },
        }

    # Add photo-z shift parameters for lenses
    lens_shifts = [0.001, 0.002, 0.001, 0.003, 0.0]
    lens_scales = [0.008, 0.007, 0.007, 0.01, 0.01]

    for i in range(min(n_bins, len(lens_shifts))):
        params[f"lens{i}_delta_z"] = {
            "ref": lens_shifts[i],
            "prior": {"dist": "norm", "loc": lens_shifts[i], "scale": lens_scales[i]},
        }

    # Add shear multiplicative bias parameters
    for i in range(min(n_bins, 4)):
        params[f"src{i}_mult_bias"] = {
            "ref": 0.001,
            "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        }

    # Add derived parameters
    for i in range(n_bins):
        params[f"TwoPoint__NumberCountsScale_lens{i}"] = {"derived": True}

    return params


def write_cobaya_config(config: Dict[str, Any], output_file: Path) -> None:
    """Write Cobaya configuration to YAML file.

    :param config: Configuration dictionary
    :param output_file: Output YAML file path
    """
    with output_file.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
