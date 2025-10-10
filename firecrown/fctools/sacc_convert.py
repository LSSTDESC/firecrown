#!/usr/bin/env python
"""Convert SACC files between FITS and HDF5 formats.

This tool reads a SACC file in either FITS or HDF5 format and writes it
in the opposite format. Format detection is automatic based on file extension,
but can be overridden with the --input-format option.
"""

import sys
from pathlib import Path

import click

try:
    import sacc
except ImportError:  # pragma: no cover
    click.echo(
        "ERROR: sacc package not found. Install it with: pip install sacc",
        err=True,
    )
    sys.exit(1)


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


@click.command()
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "Output file path. If not specified, will use input filename "
        "with new extension."
    ),
)
@click.option(
    "--input-format",
    "-f",
    type=click.Choice(["fits", "hdf5"], case_sensitive=False),
    help="Force input format (overrides automatic detection from file extension).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite output file if it exists.",
)
def main(
    input_file: Path,
    output: Path | None,
    input_format: str | None,
    overwrite: bool,
) -> None:
    r"""Convert SACC files between FITS and HDF5 formats.

    This tool reads a SACC file and converts it to the opposite format.
    Format detection is automatic based on file extension (.fits, .hdf5, .h5),
    but can be overridden with --input-format.

    Examples
    --------
    Convert FITS to HDF5 (auto-detect input, auto-generate output name)::

        python fctools/sacc_convert.py data.fits

    Convert HDF5 to FITS with specific output name::

        python fctools/sacc_convert.py data.hdf5 --output converted.fits

    Force input format when the extension is ambiguous::

        python fctools/sacc_convert.py mydata.dat --input-format fits \
            --output mydata.hdf5

    Overwrite existing output file::

        python fctools/sacc_convert.py data.fits --overwrite
    """
    # Detect or use specified input format
    if input_format:
        src_format = input_format.lower()
        click.echo(f"Using specified input format: {src_format.upper()}")
    else:
        try:
            src_format = detect_format(input_file)
            click.echo(f"Detected input format: {src_format.upper()}")
        except ValueError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)

    # Determine target format (opposite of source)
    target_format = "hdf5" if src_format == "fits" else "fits"

    # Determine output path
    output_path = determine_output_path(input_file, output, target_format)

    # Check if output exists
    if output_path.exists() and not overwrite:
        click.echo(
            f"ERROR: Output file '{output_path}' already exists. "
            "Use --overwrite to replace it.",
            err=True,
        )
        sys.exit(1)

    # Read and convert
    _read_and_convert_file(
        input_file, src_format, output_path, target_format, overwrite
    )

    # Display success info
    _display_conversion_summary(input_file, src_format, output_path, target_format)


def _read_and_convert_file(
    input_file: Path,
    src_format: str,
    output_path: Path,
    target_format: str,
    overwrite: bool,
) -> None:
    """Read input file and write to output format."""
    # Read input file
    click.echo(f"Reading {src_format.upper()} file: {input_file}")
    try:
        if src_format == "fits":
            data = sacc.Sacc.load_fits(str(input_file))
        else:  # hdf5
            data = sacc.Sacc.load_hdf5(str(input_file))
    except OSError:
        click.echo("ERROR: Failed to read input file as SACC data.")
        click.echo(
            f"The file may not be a valid SACC {src_format.upper()} file.",
            err=True,
        )
        sys.exit(1)

    # Write output file
    click.echo(f"Writing {target_format.upper()} file: {output_path}")
    try:
        if target_format == "fits":
            data.save_fits(str(output_path), overwrite=overwrite)
        else:  # hdf5
            # save_hdf5 doesn't have an overwrite parameter, so manually handle it
            if overwrite and output_path.exists():
                output_path.unlink()
            data.save_hdf5(str(output_path))
    except OSError as e:
        click.echo(f"ERROR: Failed to write SACC data to output file: {e}", err=True)
        sys.exit(1)


def _display_conversion_summary(
    input_file: Path, src_format: str, output_path: Path, target_format: str
) -> None:
    """Display conversion summary with file sizes."""
    input_size = input_file.stat().st_size
    output_size = output_path.stat().st_size

    click.echo("\n" + "=" * 60)
    click.echo("✅ Conversion successful!")
    click.echo("=" * 60)
    click.echo(f"Input:  {input_file} ({src_format.upper()}, {input_size:,} bytes)")
    click.echo(
        f"Output: {output_path} " f"({target_format.upper()}, {output_size:,} bytes)"
    )

    if output_size < input_size:
        ratio = (1 - output_size / input_size) * 100
        click.echo(f"Size reduction: {ratio:.1f}%")
    elif output_size > input_size:
        ratio = (output_size / input_size - 1) * 100
        click.echo(f"Size increase: {ratio:.1f}%")
    else:
        click.echo("Size unchanged")


if __name__ == "__main__":  # pragma: no cover
    # Click decorators inject arguments automatically from sys.argv
    main()  # pylint: disable=no-value-for-parameter
