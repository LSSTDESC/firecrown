"""Logging configuration for Firecrown."""

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

    def __post_init__(self):
        """Prepare logging."""
        self.console_io = None
        if self.log_file:
            self.console_io = self.log_file.open("w", encoding="utf-8")

        self.console = Console(file=self.console_io)

    def __del__(self):
        """Destructor to ensure file is closed."""
        if self.console_io:
            self.console_io.close()
