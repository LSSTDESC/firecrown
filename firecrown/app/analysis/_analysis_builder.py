"""Base class for Firecrown analysis builders.

Provides the abstract interface and orchestration logic for generating
complete analysis examples with data files and framework configurations.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

from typing import Annotated, ClassVar, Sequence
from abc import abstractmethod
from pathlib import Path
import dataclasses
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
import yaml
import typer
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Frameworks, Model, FrameworkCosmology, CCLCosmologySpec
from ._config_generator import get_generator, ConfigGenerator
from .. import logging
from ..sacc import SaccFormat, Transform


@dataclasses.dataclass
class AnalysisBuilder(logging.Logging):
    """Base class for Firecrown analysis builders.

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
    """Human-readable description of what this analysis demonstrates."""

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
                "(e.g., 'analysis1' creates 'analysis1.sacc')"
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

    sacc_format: Annotated[
        SaccFormat, typer.Option(help="SACC file format.", show_default=True)
    ] = SaccFormat.HDF5

    use_absolute_path: Annotated[
        bool,
        typer.Option(
            help="Use absolute file paths in configuration files",
            show_default=True,
        ),
    ] = True

    cosmology_spec: Annotated[
        Path | None,
        typer.Option(
            help="Path to cosmology specification file",
            show_default=True,
        ),
    ] = None

    def __post_init__(self) -> None:
        """Initialize and execute the complete analysis generation workflow."""
        super().__post_init__()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._spec: CCLCosmologySpec | None = None
        if self.cosmology_spec is not None:
            if not self.cosmology_spec.exists():
                raise ValueError(
                    f"Specified cosmology file does not exist: {self.cosmology_spec}"
                )
            self._spec = CCLCosmologySpec.model_validate(
                yaml.safe_load(self.cosmology_spec.read_text())
            )

        table = Table.grid(padding=(0, 1))
        table.expand = False

        # Simple helper to add labeled rows
        def add_row(label, value):
            table.add_row(f"[bold cyan]{label}[/bold cyan]", value)

        add_row("Analysis prefix:", self.prefix)

        opts_desc = self.get_options_desc()
        if opts_desc:
            table.add_row("[bold cyan]Analysis options:[/bold cyan]", "")
            for name, desc in opts_desc:
                table.add_row("  • " + name, desc)

        add_row("Framework:", self.target_framework.value)
        add_row("Description:", self.description)
        add_row("Output directory:", str(self.output_path.absolute()))

        panel = Panel.fit(table, title="[bold magenta]Analysis Builder[/bold magenta]")
        self.console.print(panel)

        # Create config generator
        generator = get_generator(
            self.target_framework,
            self.output_path,
            self.prefix,
            self.use_absolute_path,
            self.cosmology_analysis_spec(),
            self.required_cosmology(),
        )
        self._proceed_generation(generator)

    def _proceed_generation(self, generator: ConfigGenerator) -> None:
        """Execute the phased analysis generation workflow."""
        self.console.print(Rule("[bold cyan]Phase 1: Generating SACC data[/bold cyan]"))
        sacc = self.generate_sacc(self.output_path)
        if Transform.detect_format(sacc) != self.sacc_format:
            self.console.print(
                f"[yellow]Converting SACC file to target format "
                f"[bold]{self.sacc_format.upper()}[/bold][/yellow]"
            )
            transform = Transform(
                sacc_file=sacc,
                overwrite=True,
                output_format=self.sacc_format,
                quiet=True,
            )
            sacc.unlink()  # Remove original file
            sacc = transform.output_path
        generator.add_sacc(sacc)
        self.console.print(
            f"[green]OK[/green] SACC: {sacc.relative_to(self.output_path)}\n"
        )

        self.console.print(Rule("[bold cyan]Phase 2: Generating factory[/bold cyan]"))
        factory = self.generate_factory(self.output_path, sacc)
        generator.add_factory(factory)

        factory_str = (
            factory.relative_to(self.output_path)
            if isinstance(factory, Path)
            else factory
        )
        self.console.print(f"[green]OK[/green] Factory: {factory_str}\n")

        self.console.print(
            Rule("[bold cyan]Phase 3: Preparing build parameters[/bold cyan]")
        )
        build_parameters = self.get_build_parameters(sacc)
        generator.add_build_parameters(build_parameters)
        params = ", ".join(
            f"{k}={v}" for k, v in build_parameters.convert_to_basic_dict().items()
        )
        self.console.print(f"[green]OK[/green] Parameters: {params}\n")

        self.console.print(
            Rule("[bold cyan]Phase 4: Preparing model parameters[/bold cyan]")
        )
        models = self.get_models()
        generator.add_models(models)
        n_params = sum(len(m.parameters) for m in models)
        self.console.print(
            f"[green]OK[/green] Models: {len(models)} model(s), "
            f"{n_params} parameter(s)\n"
        )

        self.console.print(
            Rule(
                f"[bold cyan]Phase 5: Writing {self.target_framework.value} "
                f"configuration[/bold cyan]"
            )
        )
        generator.write_config()
        self.console.print("[green]OK[/green] Configuration written\n")

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
    def generate_factory(self, output_path: Path, sacc: Path) -> str | Path:
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

    @abstractmethod
    def required_cosmology(self) -> FrameworkCosmology:
        """Return whether the analysis requires a cosmology.

        :return: True if the analysis requires a cosmology, False otherwise
        """

    def cosmology_analysis_spec(self) -> CCLCosmologySpec:
        """Return the cosmology analysis specification.

        :return: The cosmology analysis specification
        """
        if self._spec is not None:
            return self._spec
        return CCLCosmologySpec.vanilla_lcdm()

    def get_sacc_file(self, sacc_path: Path) -> str:
        """Return the path to the SACC data file.

        Returns either an absolute or relative path based on the use_absolute_path
        setting. When use_absolute_path is True, returns the full absolute path.
        Otherwise returns just the filename.

        :param sacc_path: Path to the SACC data file
        :return: Path to the SACC file as a string, either absolute or relative
        """
        if self.use_absolute_path:
            sacc_filename = sacc_path.absolute().as_posix()
        else:
            sacc_filename = sacc_path.name
        return sacc_filename

    def get_options_desc(self) -> Sequence[tuple[str, str]]:
        """Return a description of the analysis options.

        Subclasses can override this method to provide additional
        information about the options used in the analysis.

        :return: Description of analysis options
        """
        return []
