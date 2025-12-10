"""Cobaya configuration file generator.

Generates Cobaya YAML files for cosmological parameter estimation with Firecrown.
Produces a single YAML file with theory, likelihood, parameters, and sampler config.

Cobaya natively supports both Gaussian and uniform priors, and uses CAMB for
cosmology computation. Parameters are automatically scaled (e.g., h → H0 × 100,
Omega_c → omch2 × h²) to match Cobaya/CAMB conventions.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

from typing import Any, assert_never
from pathlib import Path
import dataclasses

import yaml

import firecrown.connector.cobaya.likelihood
from firecrown.likelihood import NamedParameters
from ._types import (
    Model,
    Frameworks,
    ConfigGenerator,
    FrameworkCosmology,
    CCLCosmologySpec,
    Parameter,
    PriorGaussian,
    PriorUniform,
    get_path_str,
)

# Map CCL parameter names to Cobaya/CAMB parameter names
NAME_MAP = {
    "Omega_c": "omch2",  # Omega_c * h^2
    "Omega_b": "ombh2",  # Omega_b * h^2
    "Omega_k": "omegak",
    "T_CMB": "TCMB",
    "h": "H0",  # h * 100
    "Neff": "nnu",
    "m_nu": "mnu",
    "w0": "w",
    "wa": "wa",
    "sigma8": "sigma8",
    "A_s": "As",
    "n_s": "ns",
}


def create_config(
    factory_source: str | Path,
    build_parameters: NamedParameters,
    likelihood_name: str,
    cosmo_spec: CCLCosmologySpec,
    use_absolute_path: bool = False,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
) -> dict[str, Any]:
    """Create Cobaya configuration dictionary.

    Builds complete configuration with theory (CAMB), likelihood (Firecrown),
    parameters, and sampler sections. Uses LikelihoodConnector for external
    likelihood integration.

    :param factory_source: Path to factory file or YAML module string
    :param build_parameters: Parameters passed to likelihood factory
    :param likelihood_name: Name for likelihood in configuration
    :param cosmo_spec: Cosmology specification
    :param use_absolute_path: Use absolute paths in configuration
    :param required_cosmology: Level of cosmology computation
    :return: Configuration dictionary ready for YAML serialization
    """
    factory_source_str = get_path_str(factory_source, use_absolute_path)

    config: dict[str, Any] = {}

    use_cosmology = required_cosmology != FrameworkCosmology.NONE

    if use_cosmology:
        config["theory"] = {
            "camb": {
                "stop_at_error": True,
                "extra_args": {
                    "num_massive_neutrinos": cosmo_spec.get_num_massive_neutrinos(),
                    **(
                        cosmo_spec.extra_parameters.get_dict()
                        if cosmo_spec.extra_parameters
                        else {}
                    ),
                },
            }
        }

    config["likelihood"] = {
        likelihood_name: {
            "external": firecrown.connector.cobaya.likelihood.LikelihoodConnector,
            "firecrownIni": factory_source_str,
            "build_parameters": build_parameters.convert_to_basic_dict(),
        }
    }

    if use_cosmology:
        config["likelihood"][likelihood_name]["input_style"] = "CAMB"

    config.update(
        {
            "params": (
                _get_standard_params(
                    required_cosmology=required_cosmology,
                    cosmo_spec=cosmo_spec,
                )
                if use_cosmology
                else {}
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


def _configure_parameter(
    param: Parameter,
    param_scale: dict[str, float] | None = None,
) -> dict[str, Any] | float | None:
    """Format parameter with prior for Cobaya.

    Returns either a fixed value (if no prior) or a dict with 'ref' and 'prior'.
    Cobaya supports native Gaussian priors with 'dist': 'norm'.

    :param default_value: Default/reference parameter value
    :param prior: Prior specification (None for fixed parameters)
    :param scale: Scale factor applied to all values and bounds
    :return: Parameter configuration (dict with prior or fixed float)
    """
    param_scale = param_scale or {}
    scale = param_scale.get(param.name, 1.0)
    if not param.free:
        return param.default_value * scale

    if param.prior is None:
        return {
            "ref": param.default_value * scale,
            "prior": {
                "min": param.lower_bound * scale,
                "max": param.upper_bound * scale,
            },
        }

    match param.prior:
        case PriorGaussian():
            return {
                "ref": param.default_value * scale,
                "prior": {
                    "dist": "norm",
                    "loc": param.prior.mean * scale,
                    "scale": param.prior.sigma * scale,
                },
            }
        case PriorUniform():
            return {
                "ref": param.default_value * scale,
                "prior": {
                    "min": param.prior.lower * scale,
                    "max": param.prior.upper * scale,
                },
            }
        case _ as unreachable:
            assert_never(unreachable)


def _get_standard_params(
    required_cosmology: FrameworkCosmology,
    cosmo_spec: CCLCosmologySpec,
) -> dict[str, Any]:
    """Generate cosmological parameter configuration for Cobaya.

    Applies parameter name mapping and scaling (h → H0 x 100, Omega_c → omch2 x h²).
    Returns empty dict if no cosmology computation required.

    :param required_cosmology: Level of cosmology computation
    :param cosmo_spec: Cosmology specification with parameters and priors
    :return: Dictionary of parameter configurations
    """
    if required_cosmology == FrameworkCosmology.NONE:
        return {}
    h = cosmo_spec["h"].default_value
    h2 = h**2

    params = {}
    param_scale = {"h": 100.0, "Omega_c": h2, "Omega_b": h2}
    for param in cosmo_spec.parameters:
        name = NAME_MAP[param.name]
        params[name] = _configure_parameter(param, param_scale)

    return params


def add_models(config: dict[str, Any], models: list[Model]) -> None:
    """Add systematic/nuisance model parameters to configuration.

    Adds parameters to 'params' section with uniform priors for free parameters
    or fixed values for non-free parameters.

    :param config: Configuration dictionary (modified in-place)
    :param models: List of models with parameters
    """
    for model in models:
        for parameter in model.parameters:
            config["params"][parameter.name] = _configure_parameter(parameter)


def write_config(config: dict[str, Any], output_file: Path) -> None:
    """Write configuration dictionary to YAML file.

    Uses PyYAML with flow_style=False for readable output and
    sort_keys=False to preserve insertion order.

    :param config: Configuration dictionary
    :param output_file: Output YAML file path
    """
    with output_file.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)


@dataclasses.dataclass
class CobayaConfigGenerator(ConfigGenerator):
    """Cobaya configuration generator.

    Generates a single YAML file for Cobaya parameter estimation:

    - cobaya_{prefix}.yaml: Complete configuration with theory (CAMB),
      likelihood (Firecrown), parameters, and sampler settings
    """

    framework = Frameworks.COBAYA

    def write_config(self) -> None:
        """Write Cobaya configuration."""
        assert self.factory_source is not None
        assert self.build_parameters is not None

        cobaya_yaml = self.output_path / f"cobaya_{self.prefix}.yaml"
        likelihood_name = f"firecrown_{self.prefix}"

        cfg = create_config(
            factory_source=self.factory_source,
            build_parameters=self.build_parameters,
            use_absolute_path=self.use_absolute_path,
            likelihood_name=likelihood_name,
            cosmo_spec=self.cosmo_spec,
            required_cosmology=self.required_cosmology,
        )
        add_models(cfg, self.models)
        write_config(cfg, cobaya_yaml)
