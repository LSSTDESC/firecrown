"""Base class for Firecrown example generators.

Provides the abstract interface and orchestration logic for generating
complete analysis examples with data files and framework configurations.
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

    Orchestrates the generation of complete analysis examples through a phased workflow:

    1. Generate/download SACC data file
    2. Generate likelihood factory file
    3. Generate build parameters
    4. Generate model parameters
    5. Generate framework-specific configuration files

    Subclasses implement data-specific methods while the base class
    handles workflow orchestration and framework configuration delegation.
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
        """Initialize and execute the complete example generation workflow."""
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

        self.console.print(Rule("[bold cyan]Phase 1: Generating SACC data[/bold cyan]"))
        sacc = self.generate_sacc(self.output_path)
        generator.add_sacc(sacc)
        self.console.print(
            f"[green]✓[/green] SACC: {sacc.relative_to(self.output_path)}\n"
        )

        self.console.print(Rule("[bold cyan]Phase 2: Generating factory[/bold cyan]"))
        factory = self.generate_factory(self.output_path, sacc)
        generator.add_factory(factory)
        self.console.print(
            f"[green]✓[/green] Factory: {factory.relative_to(self.output_path)}\n"
        )

        self.console.print(
            Rule("[bold cyan]Phase 3: Preparing build parameters[/bold cyan]")
        )
        build_parameters = self.get_build_parameters(sacc)
        generator.add_build_parameters(build_parameters)
        # Here I want to print key=value pairs of the build parameters
        # in a human-readable format
        params = ", ".join(
            f"{k}={v}" for k, v in build_parameters.convert_to_basic_dict().items()
        )
        self.console.print(f"[green]✓[/green] Parameters: {params}\n")

        self.console.print(
            Rule("[bold cyan]Phase 4: Preparing model parameters[/bold cyan]")
        )
        models = self.get_models()
        generator.add_models(models)
        n_params = sum(len(m.parameters) for m in models)
        self.console.print(
            f"[green]✓[/green] Models: {len(models)} model(s), "
            f"{n_params} parameter(s)\n"
        )

        self.console.print(
            Rule(
                f"[bold cyan]Phase 5: Writing {self.target_framework.value} "
                f"configuration[/bold cyan]"
            )
        )
        generator.write_config()
        self.console.print("[green]✓[/green] Configuration written\n")

        self.console.print(
            Panel.fit("[bold green]All example files successfully created[/bold green]")
        )

    @abstractmethod
    def generate_sacc(self, output_path: Path) -> Path:
        """Generate or download the SACC data file.

        :param output_path: Directory where files should be created
        :return: Path to the generated SACC file
        """

    @abstractmethod
    def generate_factory(self, output_path: Path, sacc: Path) -> Path:
        """Generate the likelihood factory Python file.

        :param output_path: Directory where files should be created
        :param sacc: Path to the SACC data file
        :return: Path to the generated factory file
        """

    @abstractmethod
    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Create build parameters for likelihood initialization.

        :param sacc_path: Path to the SACC data file
        :return: Build parameters (typically includes sacc_file path)
        """

    @abstractmethod
    def get_models(self) -> list[Model]:
        """Define model parameters for sampling.

        :return: List of models with their parameters (priors, bounds, etc.)
        """
