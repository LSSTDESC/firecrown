"""Base class and interface for Firecrown example generators.

This module defines the abstract base class that all Firecrown examples
must inherit from. It provides a consistent interface for generating
example data and configurations through the command-line interface.
"""

from typing import Annotated, ClassVar
from abc import abstractmethod
from pathlib import Path
import dataclasses

import typer
from .. import logging


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

    def __post_init__(self) -> None:
        """Initialize example generator and create output files.

        Sets up logging, creates the output directory if needed,
        and triggers the example generation process.
        """
        super().__post_init__()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.console.print(f"[cyan]Generating example in:[/cyan] {self.output_path}")
        self.console.print()
        self.generate_sacc(self.output_path)
        self.console.print()
        self.console.print("[green]Example generation completed[/green]")

    @abstractmethod
    def generate_sacc(self, output_path: Path) -> None:
        """Generate the SACC data file and related configurations.

        This method must be implemented by each example subclass to create
        the specific data files and configurations for that analysis type.

        :param output_path: Directory where files should be created
        """
