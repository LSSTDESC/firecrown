#!/usr/bin/env python
"""Convert SACC files between FITS and HDF5 formats.

This tool reads a SACC file in either FITS or HDF5 format and writes it
in the opposite format. Format detection is automatic based on file extension,
but can be overridden with the --input-format option.
"""

import sys
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

try:
    import sacc
except ImportError:  # pragma: no cover
    console = Console()
    msg = "ERROR: sacc package not found. Install it with: pip install sacc"
    console.print(f"[bold red]{msg}[/bold red]")
    sys.exit(1)


class SaccFormat(str, Enum):
    """Enum for SACC file formats."""

    FITS = "fits"
    HDF5 = "hdf5"


def detect_format(filepath: Path) -> str:
    """Detect file format from extension.

    Args:
        filepath: Path to the file

    Returns:
        'fits' or 'hdf5'

    Raises:
        ValueError: If format cannot be detected
    """
    suffix = filepath.suffix.lower()

    if suffix == ".fits":
        return "fits"
    if suffix in (".hdf5", ".h5"):
        return "hdf5"
    raise ValueError(
        f"Cannot detect format from extension '{suffix}'. "
        "Use --input-format to specify."
    )


def determine_output_path(
    input_path: Path, output: Path | None, target_format: str
) -> Path:
    """Determine the output file path.

    Args:
        input_path: Input file path
        output: User-specified output path (optional)
        target_format: Target format ('fits' or 'hdf5')

    Returns:
        Output file path
    """
    if output:
        return output

    # Auto-generate output filename by changing extension
    stem = input_path.stem
    if target_format == "fits":
        return input_path.parent / f"{stem}.fits"
    # hdf5
    return input_path.parent / f"{stem}.hdf5"


app = typer.Typer()


@app.command()
def main(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Input SACC file path",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        writable=True,
        resolve_path=True,
        help=(
            "Output file path. "
            "If not specified, uses input filename with new extension."
        ),
    ),
    input_format: SaccFormat = typer.Option(
        None,
        "--input-format",
        "-f",
        case_sensitive=False,
        help="Force input format (overrides automatic detection).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output file if it exists."
    ),
) -> None:
    r"""Convert SACC files between FITS and HDF5 formats.

    This tool reads a SACC file and converts it to the opposite format.
    Format detection is automatic based on file extension (.fits, .hdf5, .h5),
    but can be overridden with --input-format.

    Examples::

        # Convert FITS to HDF5 (auto-detect input, auto-generate output name)
        python fctools/sacc_convert.py data.fits

        # Convert HDF5 to FITS with specific output name
        python fctools/sacc_convert.py data.hdf5 --output converted.fits

        # Force input format (useful if extension doesn't match)
        python fctools/sacc_convert.py mydata.dat --input-format fits \\
            --output mydata.hdf5

        # Overwrite existing output file
        python fctools/sacc_convert.py data.fits --overwrite
    """
    cons = Console()
    # Detect or use specified input format
    if input_format:
        src_format = input_format.value
        cons.print(f"Using specified input format: [bold]{src_format.upper()}[/bold]")
    else:
        try:
            src_format = detect_format(input_file)
            cons.print(f"Detected input format: [bold]{src_format.upper()}[/bold]")
        except ValueError as e:
            cons.print(f"[bold red]ERROR: {e}[/bold red]")
            sys.exit(1)

    # Determine target format (opposite of source)
    target_format = "hdf5" if src_format == "fits" else "fits"

    # Determine output path
    output_path = determine_output_path(input_file, output, target_format)

    # Check if output exists
    if output_path.exists() and not overwrite:
        cons.print(
            f"[bold red]ERROR: Output file '{output_path}' already exists. "
            "Use --overwrite to replace it.[/bold red]"
        )
        sys.exit(1)

    # Read and convert
    _read_and_convert_file(
        cons, input_file, src_format, output_path, target_format, overwrite
    )

    # Display success info
    _display_conversion_summary(
        cons, input_file, src_format, output_path, target_format
    )


def _read_and_convert_file(
    cons: Console,
    input_file: Path,
    src_format: str,
    output_path: Path,
    target_format: str,
    overwrite: bool,
) -> None:
    """Read input file and write to output format."""
    # Read input file
    cons.print(f"Reading {src_format.upper()} file: [cyan]{input_file}[/cyan]")
    try:
        if src_format == "fits":
            data = sacc.Sacc.load_fits(str(input_file))
        else:  # hdf5
            data = sacc.Sacc.load_hdf5(str(input_file))
    except OSError:
        cons.print(
            "[bold red]ERROR: Failed to read input file as SACC data.[/bold red]"
        )
        cons.print(
            f"The file may not be a valid SACC {src_format.upper()} file.",
        )
        sys.exit(1)

    # Write output file
    cons.print(f"Writing {target_format.upper()} file: [cyan]{output_path}[/cyan]")
    try:
        if target_format == "fits":
            data.save_fits(str(output_path), overwrite=overwrite)
        else:  # hdf5
            # save_hdf5 doesn't have an overwrite parameter, so manually handle it
            if overwrite and output_path.exists():
                output_path.unlink()
            data.save_hdf5(str(output_path))
    except OSError as e:
        cons.print(
            f"[bold red]ERROR: Failed to write SACC data to output file: {e}[/bold red]"
        )
        sys.exit(1)


def _display_conversion_summary(
    cons: Console,
    input_file: Path,
    src_format: str,
    output_path: Path,
    target_format: str,
) -> None:
    """Display conversion summary with file sizes."""
    input_size = input_file.stat().st_size
    output_size = output_path.stat().st_size

    cons.print("\n" + "=" * 60)
    cons.print("✅ [bold green]Conversion successful![/bold green]")
    cons.print("=" * 60)
    input_info = f"({src_format.upper()}, {input_size:,} bytes)"
    cons.print(f"Input:  [cyan]{input_file}[/cyan] {input_info}")
    cons.print(
        f"Output: [cyan]{output_path}[/cyan] "
        f"({target_format.upper()}, {output_size:,} bytes)"
    )

    if output_size < input_size:
        ratio = (1 - output_size / input_size) * 100
        cons.print(f"Size reduction: [bold green]{ratio:.1f}%[/bold green]")
    elif output_size > input_size:
        ratio = (output_size / input_size - 1) * 100
        cons.print(f"Size increase: [bold red]{ratio:.1f}%[/bold red]")
    else:
        cons.print("Size unchanged")


if __name__ == "__main__":  # pragma: no cover
    app()
