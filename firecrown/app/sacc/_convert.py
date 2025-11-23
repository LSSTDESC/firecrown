"""Convert command for SACC files."""

import dataclasses
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import sacc
import typer

from firecrown.app.logging import Logging


class SaccFormat(str, Enum):
    """Enum for SACC file formats."""

    FITS = "fits"
    HDF5 = "hdf5"


@dataclasses.dataclass(kw_only=True)
class Convert(Logging):
    """Convert SACC files between FITS and HDF5 formats."""

    input_file: Annotated[
        Path, typer.Argument(help="Input SACC file path", show_default=True)
    ]
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. If not specified, "
                "uses input filename with new extension."
            ),
        ),
    ] = None
    input_format: Annotated[
        SaccFormat | None,
        typer.Option(
            "--input-format",
            "-f",
            case_sensitive=False,
            help="Force input format (overrides automatic detection).",
        ),
    ] = None
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite output file if it exists.")
    ] = False

    def __post_init__(self) -> None:
        """Convert the SACC file."""
        super().__post_init__()

        if not self.input_file.exists():
            self.console.print(
                f"[bold red]ERROR: Input file not found: {self.input_file}[/bold red]"
            )
            sys.exit(1)

        self._convert_file()

    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()

        if suffix == ".fits":
            return "fits"
        if suffix in (".hdf5", ".h5"):
            return "hdf5"
        raise ValueError(
            f"Cannot detect format from extension '{suffix}'. "
            "Use --input-format to specify."
        )

    def _determine_output_path(
        self, input_path: Path, output: Path | None, target_format: str
    ) -> Path:
        """Determine the output file path."""
        if output:
            return output

        stem = input_path.stem
        if target_format == "fits":
            return input_path.parent / f"{stem}.fits"
        return input_path.parent / f"{stem}.hdf5"

    def _convert_file(self) -> None:
        """Perform the file conversion."""
        # Detect or use specified input format
        if self.input_format:
            src_format = self.input_format.value
            self.console.print(
                f"Using specified input format: [bold]{src_format.upper()}[/bold]"
            )
        else:
            try:
                src_format = self._detect_format(self.input_file)
                self.console.print(
                    f"Detected input format: [bold]{src_format.upper()}[/bold]"
                )
            except ValueError as e:
                self.console.print(f"[bold red]ERROR: {e}[/bold red]")
                sys.exit(1)

        # Determine target format
        target_format = "hdf5" if src_format == "fits" else "fits"

        # Determine output path
        output_path = self._determine_output_path(
            self.input_file, self.output, target_format
        )

        # Check if output exists
        if output_path.exists() and not self.overwrite:
            self.console.print(
                f"[bold red]ERROR: Output file '{output_path}' already exists. "
                "Use --overwrite to replace it.[/bold red]"
            )
            sys.exit(1)

        # Read and convert
        self._read_and_convert_file(src_format, output_path, target_format)

        # Display success info
        self._display_conversion_summary(src_format, output_path, target_format)

    def _read_and_convert_file(
        self, src_format: str, output_path: Path, target_format: str
    ) -> None:
        """Read input file and write to output format."""
        # Read input file
        self.console.print(
            f"Reading {src_format.upper()} file: [cyan]{self.input_file}[/cyan]"
        )
        try:
            if src_format == "fits":
                data = sacc.Sacc.load_fits(str(self.input_file))
            else:  # hdf5
                data = sacc.Sacc.load_hdf5(str(self.input_file))
        except OSError:
            self.console.print(
                "[bold red]ERROR: Failed to read input file as SACC data.[/bold red]"
            )
            self.console.print(
                f"The file may not be a valid SACC {src_format.upper()} file.",
            )
            sys.exit(1)

        # Write output file
        self.console.print(
            f"Writing {target_format.upper()} file: [cyan]{output_path}[/cyan]"
        )
        try:
            if target_format == "fits":
                data.save_fits(str(output_path), overwrite=self.overwrite)
            else:  # hdf5
                if self.overwrite and output_path.exists():
                    output_path.unlink()
                data.save_hdf5(str(output_path))
        except OSError as e:
            self.console.print(
                f"[bold red]ERROR: Failed to write SACC data to output file: "
                f"{e}[/bold red]"
            )
            sys.exit(1)

    def _display_conversion_summary(
        self, src_format: str, output_path: Path, target_format: str
    ) -> None:
        """Display conversion summary with file sizes."""
        input_size = self.input_file.stat().st_size
        output_size = output_path.stat().st_size

        self.console.print("\n" + "=" * 60)
        self.console.print("✅ [bold green]Conversion successful![/bold green]")
        self.console.print("=" * 60)
        input_info = f"({src_format.upper()}, {input_size:,} bytes)"
        self.console.print(f"Input:  [cyan]{self.input_file}[/cyan] {input_info}")
        self.console.print(
            f"Output: [cyan]{output_path}[/cyan] "
            f"({target_format.upper()}, {output_size:,} bytes)"
        )

        if output_size < input_size:
            ratio = (1 - output_size / input_size) * 100
            self.console.print(f"Size reduction: [bold green]{ratio:.1f}%[/bold green]")
        elif output_size > input_size:
            ratio = (output_size / input_size - 1) * 100
            self.console.print(f"Size increase: [bold red]{ratio:.1f}%[/bold red]")
        else:
            self.console.print("Size unchanged")
