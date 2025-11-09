"""Cobaya configuration file generation utilities.

Provides functions to create Cobaya YAML files for cosmological
parameter estimation with Firecrown likelihoods.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

from typing import Any, assert_never
from pathlib import Path
import dataclasses

import yaml

import firecrown.connector.cobaya.likelihood
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.ccl_factory import PoweSpecAmplitudeParameter
from ._types import Model, Frameworks, ConfigGenerator, FrameworkCosmology


def create_config(
    factory_path: Path,
    build_parameters: NamedParameters,
    likelihood_name: str,
    use_absolute_path: bool = False,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
    amplitude_parameter: PoweSpecAmplitudeParameter = PoweSpecAmplitudeParameter.SIGMA8,
) -> dict[str, Any]:
    """Create standard Cobaya configuration dictionary.

    :param factory_path: Path to factory file
    :param build_parameters: Likelihood build parameters
    :param likelihood_name: Likelihood name in config
    :param use_absolute_path: Use absolute paths
    :param use_cosmology: Include CAMB theory
    :return: Cobaya configuration dictionary
    """

    if use_absolute_path:
        factory_filename = factory_path.absolute().as_posix()
    else:
        factory_filename = factory_path.name

    config: dict[str, Any] = {}

    use_cosmology = required_cosmology != FrameworkCosmology.NONE

    if use_cosmology:
        config["theory"] = {
            "camb": {
                "stop_at_error": True,
                "extra_args": {"num_massive_neutrinos": 1, "halofit_version": "mead"},
            }
        }

    config["likelihood"] = {
        likelihood_name: {
            "external": firecrown.connector.cobaya.likelihood.LikelihoodConnector,
            "firecrownIni": factory_filename,
            "build_parameters": build_parameters.convert_to_basic_dict(),
        }
    }

    if use_cosmology:
        config["likelihood"][likelihood_name]["input_style"] = "CAMB"

    config.update(
        {
            "params": (
                _get_standard_params(amplitude_parameter) if use_cosmology else {}
            ),
            "sampler": {"evaluate": None},
            "stop_at_error": True,
            "output": "output",
            "packages_path": None,
            "test": False,
            "debug": False,
        }
    )

    return config


def _get_standard_params(
    amplitude_parameter: PoweSpecAmplitudeParameter,
) -> dict[str, Any]:
    """Generate standard cosmological parameter configuration.

    :return: Dictionary of parameter configurations
    """
    params = {
        # Cosmological parameters
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
    match amplitude_parameter:
        case PoweSpecAmplitudeParameter.SIGMA8:
            params["sigma8"] = {
                "prior": {"min": 0.7, "max": 1.2},
                "ref": 0.801,
                "proposal": 0.801,
            }
        case PoweSpecAmplitudeParameter.AS:
            params["As"] = {
                "prior": {"min": 0.8e-9, "max": 5.0e-9},
                "ref": 2.0e-9,
                "proposal": 2.0e-9,
            }
        case _ as unreachable:
            assert_never(unreachable)

    return params


def add_models(config: dict[str, Any], models: list[Model]) -> None:
    """Add model parameters to Cobaya configuration dictionary.

    :param config: Configuration dictionary (modified in-place)
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


def write_config(config: dict[str, Any], output_file: Path) -> None:
    """Write Cobaya configuration dictionary to YAML file.

    :param config: Configuration dictionary
    :param output_file: Output file path
    """
    with output_file.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)


@dataclasses.dataclass
class CobayaConfigGenerator(ConfigGenerator):
    """Generates Cobaya YAML configuration file.

    Creates cobaya_{prefix}.yaml with theory, likelihood, and sampler configuration.
    """

    framework = Frameworks.COBAYA

    def write_config(self) -> None:
        """Write Cobaya configuration."""
        assert self.factory_path is not None
        assert self.build_parameters is not None

        cobaya_yaml = self.output_path / f"cobaya_{self.prefix}.yaml"
        likelihood_name = f"firecrown_{self.prefix}"

        cfg = create_config(
            factory_path=self.factory_path,
            build_parameters=self.build_parameters,
            use_absolute_path=self.use_absolute_path,
            likelihood_name=likelihood_name,
            required_cosmology=self.required_cosmology,
            amplitude_parameter=self.amplitude_parameter,
        )
        add_models(cfg, self.models)
        write_config(cfg, cobaya_yaml)
