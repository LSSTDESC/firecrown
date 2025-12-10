"""Logging configuration for Firecrown."""

from io import TextIOWrapper
from typing import Annotated, Optional
import dataclasses
from pathlib import Path
import typer
from rich.console import Console


@dataclasses.dataclass(kw_only=True)
class Logging:
    """Logging configuration for Firecrown."""

    log_file: Annotated[
        Optional[Path],
        typer.Option(
            "--log-file",
            "-l",
            help="Path to the file where the log should be written.",
        ),
    ] = None

    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress console output; log only to file if --log-file is set.",
        ),
    ] = False

    def __post_init__(self):
        """Prepare logging."""
        self.console_io = None
        if self.log_file:
            self.console_io = self.log_file.open("w", encoding="utf-8")
        self.console = Console(file=self.console_io, quiet=self.quiet)

    def __del__(self):
        """Destructor to ensure file is closed."""
        # fallback: be defensive (no AttributeError)
        console_io = getattr(self, "console_io", None)
        if console_io:
            assert isinstance(console_io, TextIOWrapper)
            console_io.close()
