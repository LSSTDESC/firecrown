"""Framework-specific configuration generators.

Provides a stateful strategy pattern for generating configuration files
through phased construction: add components incrementally, then write.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
import configparser
import dataclasses

from numcosmo_py import Ncm
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Model, Frameworks
from . import _numcosmo
from . import _cosmosis
from . import _cobaya


@dataclasses.dataclass
class ConfigGenerator(ABC):
    """Abstract base for framework-specific configuration generators.

    Uses a builder pattern with phased state management:
    1. Create generator with output settings
    2. Add components: add_sacc(), add_factory(), add_build_parameters(), add_models()
    3. Write configuration: write_config()

    Subclasses implement write_config() to generate framework-specific files.
    """

    framework: ClassVar[Frameworks]
    output_path: Path
    prefix: str
    use_absolute_path: bool

    sacc_path: Path | None = None
    factory_path: Path | None = None
    build_parameters: NamedParameters | None = None
    models: list[Model] = dataclasses.field(default_factory=list)

    def add_sacc(self, sacc_path: Path) -> None:
        """Add SACC data file path to generator state.

        :param sacc_path: Path to SACC data file
        """
        self.sacc_path = sacc_path

    def add_factory(self, factory_path: Path) -> None:
        """Add factory file path to generator state.

        :param factory_path: Path to likelihood factory Python file
        """
        self.factory_path = factory_path

    def add_build_parameters(self, build_parameters: NamedParameters) -> None:
        """Add build parameters to generator state.

        :param build_parameters: Parameters for likelihood initialization
        """
        self.build_parameters = build_parameters

    def add_models(self, models: list[Model]) -> None:
        """Add model parameters to generator state.

        :param models: List of models with sampling parameters
        """
        self.models = models

    @abstractmethod
    def write_config(self) -> None:
        """Write framework-specific configuration files using accumulated state."""


class CosmosisConfigGenerator(ConfigGenerator):
    """Generates CosmoSIS .ini configuration files.

    Creates two files:
    - cosmosis_{prefix}.ini: Pipeline configuration
    - cosmosis_{prefix}_values.ini: Parameter values and priors
    """

    framework = Frameworks.COSMOSIS

    def write_config(self) -> None:
        """Write CosmoSIS configuration."""
        assert self.factory_path is not None
        assert self.build_parameters is not None

        cosmosis_ini = self.output_path / f"cosmosis_{self.prefix}.ini"
        values_ini = self.output_path / f"cosmosis_{self.prefix}_values.ini"

        cfg = _cosmosis.create_standard_cosmosis_config(
            prefix=self.prefix,
            factory_path=self.factory_path,
            build_parameters=self.build_parameters,
            values_path=values_ini,
            output_path=self.output_path,
            model_list=[f"firecrown_{self.prefix}"],
            use_absolute_path=self.use_absolute_path,
        )

        values_cfg = _cosmosis.create_standard_values_config(self.models)

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
        _cosmosis.add_comment_block(cfg, section, comment)
        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)


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

        cfg = _cobaya.create_standard_cobaya_config(
            factory_path=self.factory_path,
            build_parameters=self.build_parameters,
            use_absolute_path=self.use_absolute_path,
            likelihood_name="firecrown_likelihood",
        )

        _cobaya.add_models_to_cobaya_config(cfg, self.models)
        _cobaya.write_cobaya_config(cfg, cobaya_yaml)


class NumCosmoConfigGenerator(ConfigGenerator):
    """Generates NumCosmo YAML configuration files.

    Creates two files:
    - numcosmo_{prefix}.yaml: Experiment configuration
    - numcosmo_{prefix}.builders.yaml: Model builders
    """

    framework = Frameworks.NUMCOSMO

    def write_config(self) -> None:
        """Write NumCosmo configuration."""
        assert self.factory_path is not None
        assert self.build_parameters is not None

        Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

        config = _numcosmo.create_standard_numcosmo_config(
            factory_path=self.factory_path,
            build_parameters=self.build_parameters,
            model_list=[f"firecrown_{self.prefix}"],
            use_absolute_path=self.use_absolute_path,
        )

        model_builders = _numcosmo.add_models_to_numcosmo_config(config, self.models)

        numcosmo_yaml = self.output_path / f"numcosmo_{self.prefix}.yaml"
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


def get_generator(
    framework: Frameworks, output_path: Path, prefix: str, use_absolute_path: bool
) -> ConfigGenerator:
    """Factory function to create framework-specific configuration generator.

    :param framework: Target framework (cosmosis, cobaya, numcosmo)
    :param output_path: Directory where configuration files will be written
    :param prefix: Prefix for generated filenames
    :param use_absolute_path: Use absolute paths in configuration files
    :return: Initialized configuration generator
    :raises ValueError: If framework is not supported
    """
    generator_class = _GENERATORS.get(framework)
    if generator_class is None:
        raise ValueError(f"Unsupported framework: {framework}")
    return generator_class(
        output_path=output_path, prefix=prefix, use_absolute_path=use_absolute_path
    )
