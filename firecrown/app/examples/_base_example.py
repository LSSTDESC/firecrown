"""Base class and interface for Firecrown example generators.

This module defines the abstract base class that all Firecrown examples
must inherit from. It provides a consistent interface for generating
example data and configurations through the command-line interface.
"""

from typing import Annotated, ClassVar
from abc import abstractmethod
from pathlib import Path
import dataclasses
from rich.panel import Panel
from rich.rule import Rule

import typer
from numcosmo_py import Ncm
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Frameworks, Model
from .. import logging
from . import _cobaya
from . import _cosmosis
from . import _numcosmo


@dataclasses.dataclass
class Example(logging.Logging):
    """Base class for Firecrown example generators.

    This abstract base class defines the interface that all example generators
    must implement. It handles common functionality like output directory creation
    and logging setup, while requiring subclasses to implement the specific
    data generation logic.

    Each example generator creates/downloads data and configuration files
    that demonstrate a particular type of cosmological analysis.
    """

    description: ClassVar[str]
    """Human-readable description of what this example demonstrates."""

    output_path: Annotated[
        Path,
        typer.Argument(
            help="Directory where example files will be generated", show_default=True
        ),
    ]

    prefix: Annotated[
        str,
        typer.Option(
            help=(
                "Prefix for generated filenames "
                "(e.g., 'cosmic_shear' creates 'cosmic_shear.sacc')"
            ),
            show_default=True,
        ),
    ]

    target_framework: Annotated[
        Frameworks,
        typer.Option(
            help="Framework to generate example for",
            show_default=True,
            case_sensitive=False,
        ),
    ] = Frameworks.COSMOSIS

    use_absolute_path: Annotated[
        bool,
        typer.Option(
            help="Use absolute file paths in configuration files",
            show_default=True,
        ),
    ] = True

    def __post_init__(self) -> None:
        """Initialize example generator and create output files.

        Sets up logging, creates the output directory if needed,
        and triggers the example generation process.
        """
        super().__post_init__()
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.console.print(
            Panel.fit(
                f"[bold cyan]Output directory:[/bold cyan] "
                f"{self.output_path.absolute()}"
            )
        )
        self.console.print(Rule("[bold cyan]Generating example data[/bold cyan]"))
        sacc = self.generate_sacc(self.output_path)
        self.console.print("[green]Example data generated[/green]\n")

        self.console.print(Rule("[bold cyan]Generating factory[/bold cyan]"))
        factory = self.generate_factory(self.output_path, sacc)
        self.console.print("[green]Factory generated[/green]\n")

        self.console.print(Rule("[bold cyan]Generating build parameters[/bold cyan]"))
        build_parameters = self.get_build_parameters(sacc)
        self.console.print("[green]Build parameters generated[/green]\n")

        self.console.print(Rule("[bold cyan]Generating configuration[/bold cyan]"))
        self.generate_config(self.output_path, factory, build_parameters)
        self.console.print("[green]Configuration generated[/green]\n")

        self.console.print(
            Panel.fit("[bold green]All example files successfully created[/bold green]")
        )

    @abstractmethod
    def generate_sacc(self, output_path: Path) -> Path:
        """Generate the SACC data file and related configurations.

        This method must be implemented by each example subclass to create
        the specific data files and configurations for that analysis type.

        :param output_path: Directory where files should be created
        """

    @abstractmethod
    def generate_factory(self, output_path: Path, sacc: Path) -> Path:
        """Generate the factory file and related configurations.

        This method must be implemented by each example subclass to create
        the specific factory files and configurations for that analysis type.

        :param output_path: Directory where files should be created
        :return: The string to be used as the factory in the configuration
        """

    @abstractmethod
    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Generate example build parameters.

        This method must be implemented by each example subclass to create
        the specific build parameters for that analysis type.

        :return: A NamedParameters of build parameters
        """

    @abstractmethod
    def get_models(
        self,
    ) -> list[Model]:
        """Generate example model parameters.

        This method must be implemented by each example subclass to create
        the specific model parameters for that analysis type.

        :return: A list of models with associated parameters
        """

    def generate_cosmosis_config(
        self, output_path: Path, factory_path: Path, build_parameters: NamedParameters
    ) -> None:
        """Generate CosmoSIS configuration files."""
        cosmosis_ini = output_path / f"cosmosis_{self.prefix}.ini"
        values_ini = output_path / f"cosmosis_{self.prefix}_values.ini"
        model_name = f"firecrown_{self.prefix}"

        cfg = _cosmosis.create_standard_cosmosis_config(
            prefix=self.prefix,
            factory_path=factory_path,
            build_parameters=build_parameters,
            values_path=values_ini,
            output_path=output_path,
            model_list=[model_name],
            use_absolute_path=self.use_absolute_path,
        )

        values_cfg = _cosmosis.create_standard_values_config(self.get_models())

        with values_ini.open("w") as fp:
            values_cfg.write(fp)

        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)

    def generate_cobaya_config(
        self, output_path: Path, factory_path: Path, build_parameters: NamedParameters
    ) -> None:
        """Generate Cobaya configuration files."""

        cobaya_yaml = output_path / f"cobaya_{self.prefix}.yaml"

        cfg = _cobaya.create_standard_cobaya_config(
            factory_path=factory_path,
            build_parameters=build_parameters,
            use_absolute_path=self.use_absolute_path,
            likelihood_name="firecrown_likelihood",
        )

        _cobaya.add_models_to_cobaya_config(cfg, self.get_models())
        _cobaya.write_cobaya_config(cfg, cobaya_yaml)

    def generate_numcosmo_config(
        self, output_path: Path, factory_path: Path, build_parameters: NamedParameters
    ) -> None:
        """Generate NumCosmo configuration files."""
        Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

        model_name = f"firecrown_{self.prefix}"
        model_list = [model_name]

        config = _numcosmo.create_standard_numcosmo_config(
            factory_path=factory_path,
            build_parameters=build_parameters,
            model_list=model_list,
            use_absolute_path=self.use_absolute_path,
        )

        model_builders = _numcosmo.add_models_to_numcosmo_config(
            config, self.get_models()
        )

        numcosmo_yaml = output_path / f"numcosmo_{self.prefix}.yaml"
        builders_file = numcosmo_yaml.with_suffix(".builders.yaml")

        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        ser.dict_str_to_yaml_file(config, numcosmo_yaml.as_posix())
        ser.dict_str_to_yaml_file(model_builders, builders_file.as_posix())

    def generate_config(
        self, output_path: Path, factory: Path, build_params: NamedParameters
    ) -> None:
        """Generate the configuration file and related configurations.

        This method must be implemented by each example subclass to create
        the specific configuration files and configurations for that analysis type.

        :param output_path: Directory where files should be created
        """
        try:
            match self.target_framework:
                case Frameworks.COSMOSIS:
                    self.generate_cosmosis_config(output_path, factory, build_params)
                case Frameworks.COBAYA:
                    self.generate_cobaya_config(output_path, factory, build_params)
                case Frameworks.NUMCOSMO:
                    self.generate_numcosmo_config(output_path, factory, build_params)
        except NotImplementedError as e:
            # Typer will format this nicely for the CLI user
            raise typer.BadParameter(
                f"Cannot generate config for {self.target_framework.value}: {e}"
            )
