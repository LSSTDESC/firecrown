"""Load command for SACC files."""

import dataclasses
import contextlib
import io
import warnings
from pathlib import Path
from typing import Annotated

import typer

from firecrown.likelihood import factories
from firecrown.app.logging import Logging


@dataclasses.dataclass(kw_only=True)
class Load(Logging):
    """Load and summarize a SACC file."""

    sacc_file: Annotated[
        Path, typer.Argument(help="Path to the SACC file.", show_default=True)
    ]
    allow_mixed_types: Annotated[
        bool,
        typer.Option(
            "--allow-mixed-types",
            help=(
                "Allow measurements with types from different sets "
                "(e.g., galaxy source + lens types)."
            ),
        ),
    ] = False

    def __post_init__(self) -> None:
        """Load and display metadata from the SACC file."""
        super().__post_init__()
        self._load_sacc_file()

    def _load_sacc_file(self) -> None:
        """Load the SACC file, with error handling for missing or unreadable files."""
        self.console.rule("[bold blue]Loading SACC file[/bold blue]")
        self.console.print(f"[cyan]File:[/cyan] {self.sacc_file}")
        self.console.print(f"[cyan]Allow mixed types:[/cyan] {self.allow_mixed_types}")
        try:
            if not self.sacc_file.exists():
                raise typer.BadParameter(f"SACC file not found: {self.sacc_file}")

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                self.sacc_data = factories.load_sacc_data(self.sacc_file.as_posix())
        except Exception as e:
            self.console.print(f"[bold red]Failed to load SACC file:[/bold red] {e}")
            raise
