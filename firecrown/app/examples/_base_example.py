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
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Frameworks, Model
from ._config_generator import get_generator
from .. import logging


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

        # Create config generator
        generator = get_generator(
            self.target_framework, self.output_path, self.prefix, self.use_absolute_path
        )

        self.console.print(Rule("[bold cyan]Generating example data[/bold cyan]"))
        sacc = self.generate_sacc(self.output_path)
        generator.add_sacc(sacc)
        self.console.print("[green]Example data generated[/green]\n")

        self.console.print(Rule("[bold cyan]Generating factory[/bold cyan]"))
        factory = self.generate_factory(self.output_path, sacc)
        generator.add_factory(factory)
        self.console.print("[green]Factory generated[/green]\n")

        self.console.print(Rule("[bold cyan]Generating build parameters[/bold cyan]"))
        build_parameters = self.get_build_parameters(sacc)
        generator.add_build_parameters(build_parameters)
        self.console.print("[green]Build parameters generated[/green]\n")

        self.console.print(Rule("[bold cyan]Generating Firecrown models[/bold cyan]"))
        models = self.get_models()
        generator.add_models(models)
        self.console.print("[green]Firecrown models generated[/green]\n")

        self.console.print(Rule("[bold cyan]Generating configuration[/bold cyan]"))
        generator.write_config()
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
