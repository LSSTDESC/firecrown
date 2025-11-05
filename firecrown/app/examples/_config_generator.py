"""Configuration generator interface and implementations for different frameworks.

This module provides a unified interface for generating framework-specific
configuration files, using a strategy pattern to encapsulate framework details.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
import configparser

from numcosmo_py import Ncm
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Model, Frameworks
from . import _cosmosis
from . import _cobaya
from . import _numcosmo


class ConfigGenerator(ABC):
    """Abstract base class for framework-specific configuration generators."""

    framework: ClassVar[Frameworks]

    @abstractmethod
    def generate(
        self,
        output_path: Path,
        factory_path: Path,
        build_parameters: NamedParameters,
        models: list[Model],
        prefix: str,
        use_absolute_path: bool,
    ) -> None:
        """Generate configuration files for the framework.

        :param output_path: Directory where files should be created
        :param factory_path: Path to the factory file
        :param build_parameters: Build parameters for the likelihood
        :param models: List of models with parameters
        :param prefix: Prefix for generated filenames
        :param use_absolute_path: Whether to use absolute paths
        """


class CosmosisConfigGenerator(ConfigGenerator):
    """CosmoSIS configuration generator."""

    framework = Frameworks.COSMOSIS

    def generate(
        self,
        output_path: Path,
        factory_path: Path,
        build_parameters: NamedParameters,
        models: list[Model],
        prefix: str,
        use_absolute_path: bool,
    ) -> None:

        cosmosis_ini = output_path / f"cosmosis_{prefix}.ini"
        values_ini = output_path / f"cosmosis_{prefix}_values.ini"

        cfg = _cosmosis.create_standard_cosmosis_config(
            prefix=prefix,
            factory_path=factory_path,
            build_parameters=build_parameters,
            values_path=values_ini,
            output_path=output_path,
            model_list=[f"firecrown_{prefix}"],
            use_absolute_path=use_absolute_path,
        )

        values_cfg = _cosmosis.create_standard_values_config(models)

        with values_ini.open("w") as fp:
            values_cfg.write(fp)
        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)

    def customize_config(
        self, config: configparser.ConfigParser, section: str, comment: str
    ) -> None:
        """Allow subclasses to customize the config after generation."""

        _cosmosis.add_comment_block(config, section, comment)


class CobayaConfigGenerator(ConfigGenerator):
    """Cobaya configuration generator."""

    framework = Frameworks.COBAYA

    def generate(
        self,
        output_path: Path,
        factory_path: Path,
        build_parameters: NamedParameters,
        models: list[Model],
        prefix: str,
        use_absolute_path: bool,
    ) -> None:
        cobaya_yaml = output_path / f"cobaya_{prefix}.yaml"

        cfg = _cobaya.create_standard_cobaya_config(
            factory_path=factory_path,
            build_parameters=build_parameters,
            use_absolute_path=use_absolute_path,
            likelihood_name="firecrown_likelihood",
        )

        _cobaya.add_models_to_cobaya_config(cfg, models)
        _cobaya.write_cobaya_config(cfg, cobaya_yaml)


class NumCosmoConfigGenerator(ConfigGenerator):
    """NumCosmo configuration generator."""

    framework = Frameworks.NUMCOSMO

    def generate(
        self,
        output_path: Path,
        factory_path: Path,
        build_parameters: NamedParameters,
        models: list[Model],
        prefix: str,
        use_absolute_path: bool,
    ) -> None:
        Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

        config = _numcosmo.create_standard_numcosmo_config(
            factory_path=factory_path,
            build_parameters=build_parameters,
            model_list=[f"firecrown_{prefix}"],
            use_absolute_path=use_absolute_path,
        )

        model_builders = _numcosmo.add_models_to_numcosmo_config(config, models)

        numcosmo_yaml = output_path / f"numcosmo_{prefix}.yaml"
        builders_file = numcosmo_yaml.with_suffix(".builders.yaml")

        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        ser.dict_str_to_yaml_file(config, numcosmo_yaml.as_posix())
        ser.dict_str_to_yaml_file(model_builders, builders_file.as_posix())


# Registry of available generators
_GENERATORS: dict[Frameworks, type[ConfigGenerator]] = {
    Frameworks.COSMOSIS: CosmosisConfigGenerator,
    Frameworks.COBAYA: CobayaConfigGenerator,
    Frameworks.NUMCOSMO: NumCosmoConfigGenerator,
}


def get_generator(framework: Frameworks) -> ConfigGenerator:
    """Get the appropriate config generator for a framework.

    :param framework: Target framework
    :return: Config generator instance
    :raises ValueError: If framework is not supported
    """
    generator_class = _GENERATORS.get(framework)
    if generator_class is None:
        raise ValueError(f"Unsupported framework: {framework}")
    return generator_class()
