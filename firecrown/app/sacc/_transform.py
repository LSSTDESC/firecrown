"""Transform command for SACC files."""

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
class Transform(Logging):
    """Transform SACC files by updating internal representation.

    This command reads a SACC file and writes it back, which updates the internal
    representation of older SACC files to the current format. Optionally converts
    between FITS and HDF5 formats.
    """

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
                "uses input filename with new extension (if format changes)."
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
    output_format: Annotated[
        SaccFormat | None,
        typer.Option(
            "--output-format",
            "-t",
            case_sensitive=False,
            help="Output format. If not specified, uses input format.",
        ),
    ] = None
    fix_ordering: Annotated[
        bool,
        typer.Option(
            "--fix-ordering",
            help="Fix tracer ordering issues (not yet implemented).",
        ),
    ] = False
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite output file if it exists.")
    ] = False

    def __post_init__(self) -> None:
        """Transform the SACC file."""
        super().__post_init__()

        if not self.input_file.exists():
            self.console.print(
                f"[bold red]ERROR: Input file not found: {self.input_file}[/bold red]"
            )
            sys.exit(1)

        if self.fix_ordering:
            self.console.print(
                "[bold yellow]WARNING: --fix-ordering is not yet implemented[/bold yellow]"
            )

        self._transform_file()

    def _detect_format(self, filepath: Path) -> SaccFormat:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()

        match suffix:
            case ".fits":
                return SaccFormat.FITS
            case ".hdf5" | ".h5":
                return SaccFormat.HDF5
            case _:
                try:
                    sacc.Sacc.load_fits(str(filepath))
                    return SaccFormat.FITS
                except OSError:
                    pass
                try:
                    sacc.Sacc.load_hdf5(str(filepath))
                    return SaccFormat.HDF5
                except OSError:
                    pass
                raise ValueError(
                    f"Cannot detect format from extension '{suffix}'. "
                    "Use --input-format to specify."
                )

    def _determine_output_path(
        self, input_path: Path, output: Path | None, target_format: SaccFormat
    ) -> Path:
        """Determine the output file path."""
        if output:
            return output

        # If format is the same, use the input path
        stem = input_path.stem
        match target_format:
            case SaccFormat.FITS:
                return input_path.parent / f"{stem}.fits"
            case SaccFormat.HDF5:
                return input_path.parent / f"{stem}.hdf5"
            case _:
                raise ValueError(f"Unknown target format: {target_format}")

    def _transform_file(self) -> None:
        """Perform the file transformation."""
        # Detect or use specified input format
        if self.input_format:
            src_format = self.input_format
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

        # Determine target format (same as input if not specified)
        target_format = self.output_format if self.output_format else src_format

        # Determine output path
        output_path = self._determine_output_path(
            self.input_file, self.output, target_format
        )

        # Check if output would overwrite input without --overwrite
        if output_path == self.input_file and not self.overwrite:
            self.console.print(
                f"[bold red]ERROR: Output path '{output_path}' is the same as input. "
                "Use --overwrite to update in place.[/bold red]"
            )
            sys.exit(1)

        # Check if output exists
        if (
            output_path.exists()
            and output_path != self.input_file
            and not self.overwrite
        ):
            self.console.print(
                f"[bold red]ERROR: Output file '{output_path}' already exists. "
                "Use --overwrite to replace it.[/bold red]"
            )
            sys.exit(1)

        # Read and transform
        self._read_and_transform_file(src_format, output_path, target_format)

        # Display success info
        self._display_transform_summary(src_format, output_path, target_format)

    def _read_and_transform_file(
        self, src_format: str, output_path: Path, target_format: SaccFormat
    ) -> None:
        """Read input file and write to output format."""
        # Read input file
        self.console.print(
            f"Reading {src_format.upper()} file: [cyan]{self.input_file}[/cyan]"
        )
        try:
            match src_format:
                case SaccFormat.FITS | "fits":
                    data = sacc.Sacc.load_fits(str(self.input_file))
                case SaccFormat.HDF5 | "hdf5":
                    data = sacc.Sacc.load_hdf5(str(self.input_file))
                case _:
                    raise ValueError(f"Unknown input format: {src_format}")
        except OSError:
            self.console.print(
                "[bold red]ERROR: Failed to read input file as SACC data.[/bold red]"
            )
            self.console.print(
                f"The file may not be a valid SACC {src_format} file.",
            )
            sys.exit(1)

        # Write output file
        self.console.print(f"Writing {target_format} file: [cyan]{output_path}[/cyan]")
        try:
            match target_format:
                case SaccFormat.FITS:
                    data.save_fits(str(output_path), overwrite=self.overwrite)
                case SaccFormat.HDF5:
                    if self.overwrite and output_path.exists():
                        output_path.unlink()
                    data.save_hdf5(str(output_path))
                case _:
                    raise ValueError(f"Unknown output format: {target_format}")
        except OSError as e:
            self.console.print(
                f"[bold red]ERROR: Failed to write SACC data to output file: "
                f"{e}[/bold red]"
            )
            sys.exit(1)

    def _display_transform_summary(
        self, src_format: str, output_path: Path, target_format: SaccFormat
    ) -> None:
        """Display transformation summary with file sizes."""
        input_size = self.input_file.stat().st_size
        output_size = output_path.stat().st_size

        self.console.print("\n" + "=" * 60)
        self.console.print("✅ [bold green]Transformation successful![/bold green]")
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
