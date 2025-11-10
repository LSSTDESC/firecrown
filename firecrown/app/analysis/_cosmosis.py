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
import firecrown
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.ccl_factory import PoweSpecAmplitudeParameter

from ._types import Model, Frameworks, ConfigGenerator, FrameworkCosmology, get_path_str


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
    output_path: Path,
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


def add_cosmological_parameters(
    config: configparser.ConfigParser,
    amplitude_parameter: PoweSpecAmplitudeParameter = PoweSpecAmplitudeParameter.SIGMA8,
) -> None:
    """Add cosmological parameters section to CosmoSIS values config.

    :param config: ConfigParser object (modified in-place)
    :param amplitude_parameter: Power spectrum amplitude parameter
    """
    section = "cosmological_parameters"
    config.add_section(section)

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
    config.set(section, "h0", "0.682")
    config.set(section, "w", "-1.0")
    config.set(section, "wa", "0.0")

    match amplitude_parameter:
        case PoweSpecAmplitudeParameter.SIGMA8:
            config.set(section, "sigma_8", "0.801")
        case PoweSpecAmplitudeParameter.AS:
            config.set(section, "A_s", "2.0e-9")
        case _ as unreachable:
            assert_never(unreachable)


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
    models: list[Model] | None = None,
    amplitude_parameter: PoweSpecAmplitudeParameter = PoweSpecAmplitudeParameter.SIGMA8,
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
        add_cosmological_parameters(config, amplitude_parameter)

    if models:
        add_firecrown_models(config, models)

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

        cfg = create_config(
            prefix=self.prefix,
            factory_source=self.factory_source,
            build_parameters=self.build_parameters,
            values_path=values_ini,
            output_path=self.output_path,
            use_absolute_path=self.use_absolute_path,
            required_cosmology=self.required_cosmology,
        )
        add_models(cfg, self.models)

        values_cfg = create_values_config(
            self.models, self.amplitude_parameter, self.required_cosmology
        )

        with values_ini.open("w") as fp:
            values_cfg.write(fp)
        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)

    def customize_config(self, section: str, comment: str) -> None:
        """Add custom comment block to configuration section.

        :param section: Configuration section name
        :param comment: Comment text to add
        """
        cosmosis_ini = self.output_path / f"cosmosis_{self.prefix}.ini"
        cfg = configparser.ConfigParser()
        cfg.read(cosmosis_ini)
        add_comment_block(cfg, section, comment)
        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)
