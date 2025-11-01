"""Base class and interface for Firecrown example generators.

This module defines the abstract base class that all Firecrown examples
must inherit from. It provides a consistent interface for generating
example data and configurations through the command-line interface.
"""

from typing import Annotated, ClassVar
from enum import StrEnum
from abc import abstractmethod
from pathlib import Path
import dataclasses

import typer
from .. import logging


class Frameworks(StrEnum):
    """Supported frameworks for example generation."""

    COBAYA = "cobaya"
    COSMOSIS = "cosmosis"
    NUMCOSMO = "numcosmo"


@dataclasses.dataclass
class Example(logging.Logging):
    """Base class for Firecrown example generators.

    This abstract base class defines the interface that all example generators
    must implement. It handles common functionality like output directory creation
    and logging setup, while requiring subclasses to implement the specific
    data generation logic.

    Each example generator creates synthetic data and configuration files
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

    def __post_init__(self) -> None:
        """Initialize example generator and create output files.

        Sets up logging, creates the output directory if needed,
        and triggers the example generation process.
        """
        super().__post_init__()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.console.print(f"[cyan]Generating example in:[/cyan] {self.output_path}")
        self.console.print()
        sacc = self.generate_sacc(self.output_path)
        self.console.print()
        self.console.print("[green]Example generation completed[/green]")

        self.console.print()
        self.console.print(
            f"[cyan]Generating example factory in:[/cyan] {self.output_path}"
        )
        factory = self.generate_factory(self.output_path, sacc)
        self.console.print()
        self.console.print("[green]Example factory completed[/green]")

        self.console.print()
        self.console.print(
            f"[cyan]Generating example configuration in:[/cyan] {self.output_path}"
        )
        self.generate_config(self.output_path, sacc, factory)
        self.console.print()
        self.console.print("[green]Example configuration completed[/green]")

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

    def generate_cosmosis_config(
        self, _output_path: Path, _sacc: Path, _factory: Path
    ) -> None:
        """Generate example configuration file for Cosmosis."""
        err = NotImplementedError(
            f"{self.__class__.__name__} does not support CosmoSIS configuration."
        )
        raise err

    def generate_cobaya_config(
        self, _output_path: Path, _sacc: Path, _factory: Path
    ) -> None:
        """Generate example configuration file for Cobaya."""
        err = NotImplementedError(
            f"{self.__class__.__name__} does not support Cobaya configuration."
        )
        raise err

    def generate_numcosmo_config(
        self, _output_path: Path, _sacc: Path, _factory: Path
    ) -> None:
        """Generate example configuration file for NumCosmo."""
        err = NotImplementedError(
            f"{self.__class__.__name__} does not support NumCosmo configuration."
        )
        raise err

    def generate_config(self, output_path: Path, sacc: Path, factory: Path) -> None:
        """Generate the configuration file and related configurations.

        This method must be implemented by each example subclass to create
        the specific configuration files and configurations for that analysis type.

        :param output_path: Directory where files should be created
        """

        try:
            match self.target_framework:
                case Frameworks.COSMOSIS:
                    self.generate_cosmosis_config(output_path, sacc, factory)
                case Frameworks.COBAYA:
                    self.generate_cobaya_config(output_path, sacc, factory)
                case Frameworks.NUMCOSMO:
                    self.generate_numcosmo_config(output_path, sacc, factory)
        except NotImplementedError as e:
            # Typer will format this nicely for the CLI user
            raise typer.BadParameter(
                f"Cannot generate config for {self.target_framework.value}: {e}"
            )
