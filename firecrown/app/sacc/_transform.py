"""Transform command for SACC files."""

import dataclasses
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, assert_never

import sacc
import typer

from firecrown.metadata_functions import (
    extract_all_real_metadata_indices,
    extract_all_harmonic_metadata_indices,
)
from ._load import Load


class SaccFormat(str, Enum):
    """Enum for SACC file formats."""

    FITS = "fits"
    HDF5 = "hdf5"


@dataclasses.dataclass(kw_only=True)
class Transform(Load):
    """Transform SACC files by updating internal representation and fixing issues.

    This command reads a SACC file and writes it back, which updates the internal
    representation of older SACC files to the current format. Optionally converts
    between FITS and HDF5 formats and fixes tracer ordering issues.

    The primary use cases are:

    1. Updating legacy SACC files to the current internal format
    2. Converting between FITS and HDF5 formats
    3. Fixing tracer ordering violations that don't follow SACC naming conventions

    CLI Usage::

        # Update internal format (in-place)
        firecrown sacc transform data.sacc --overwrite

        # Fix tracer ordering issues
        firecrown sacc transform data.sacc --fix-ordering --overwrite

        # Convert FITS to HDF5
        firecrown sacc transform data.fits --output-format hdf5 --output data.h5

        # Fix ordering and convert format
        firecrown sacc transform data.fits --fix-ordering --output-format hdf5 -o \
data.h5

    Python Usage::

        from pathlib import Path
        from firecrown.app.sacc import Transform

        # Fix ordering and update file
        Transform(
            sacc_file=Path("data.sacc"),
            fix_ordering=True,
            overwrite=True
        )

    :param sacc_file: Path to the input SACC file (inherited from Load)
    :type sacc_file: Path
    :param output: Output file path. If not specified, uses input filename with new
        extension if format changes.
    :type output: Path | None
    :param input_format: Force input format (overrides automatic detection).
    :type input_format: SaccFormat | None
    :param output_format: Output format. If not specified, uses input format.
    :type output_format: SaccFormat | None
    :param fix_ordering: Fix tracer ordering issues according to SACC naming
        conventions.
    :type fix_ordering: bool
    :param overwrite: Overwrite output file if it exists.
    :type overwrite: bool
    :param allow_mixed_types: Allow tracers with mixed measurement types
        (inherited from Load).
    :type allow_mixed_types: bool

    .. seealso::
        :class:`View` - For inspecting SACC files and detecting issues

        ``docs/sacc_usage.rst`` - Comprehensive guide to SACC conventions
    """

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
            help="Fix tracer ordering issues according to SACC naming conventions.",
        ),
    ] = False
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite output file if it exists.")
    ] = False

    def __post_init__(self) -> None:
        """Transform the SACC file."""
        # Do not call Load's _load_sacc_file, but keep console and error handling
        super().__post_init__()

        if not self.sacc_file.exists():
            self.console.print(
                f"[bold red]ERROR: Input file not found: {self.sacc_file}[/bold red]"
            )
            sys.exit(1)

        # Prepare transformation: determine formats and output path, check for errors
        output_path, src_format, target_format = self._prepare_transform()
        # Read SACC data
        sacc_data = self._read_sacc_data(src_format)
        # Optionally fix ordering
        if self.fix_ordering:
            self._fix_ordering(sacc_data)
        # Write SACC data
        self._write_sacc_data(sacc_data, output_path, target_format)
        # Display success info
        self._display_transform_summary(src_format, output_path, target_format)
        self.output_path = output_path
        self.src_format = src_format
        self.target_format = target_format

    def _load_sacc_file(self) -> None:
        """Override Load's file loader to do nothing for Transform."""

    @staticmethod
    def detect_format(filepath: Path) -> SaccFormat:
        """Detect file format from extension or file contents."""
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
            case _ as unreachable:
                assert_never(unreachable)

    def _prepare_transform(self) -> tuple[Path, SaccFormat, SaccFormat]:
        """Prepare transformation.

        Determine formats, output path, and check for errors.
        """
        # Detect or use specified input format
        if self.input_format:
            src_format = self.input_format
            self.console.print(
                f"Using specified input format: [bold]{src_format.upper()}[/bold]"
            )
        else:
            try:
                src_format = self.detect_format(self.sacc_file)
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
            self.sacc_file, self.output, target_format
        )

        # Check if output would overwrite input without --overwrite
        if output_path == self.sacc_file and not self.overwrite:
            self.console.print(
                f"[bold red]ERROR: Output path '{output_path}' is the same as input. "
                "Use --overwrite to update in place.[/bold red]"
            )
            sys.exit(1)

        # Check if output exists
        if (
            output_path.exists()
            and output_path != self.sacc_file
            and not self.overwrite
        ):
            self.console.print(
                f"[bold red]ERROR: Output file '{output_path}' already exists. "
                "Use --overwrite to replace it.[/bold red]"
            )
            sys.exit(1)

        return output_path, src_format, target_format

    def _read_sacc_data(self, src_format: str):
        """Read input file and return SACC data object."""
        self.console.print(
            f"Reading {src_format.upper()} file: [cyan]{self.sacc_file}[/cyan]"
        )
        try:
            match src_format:
                case SaccFormat.FITS | "fits":
                    return sacc.Sacc.load_fits(str(self.sacc_file))
                case SaccFormat.HDF5 | "hdf5":
                    return sacc.Sacc.load_hdf5(str(self.sacc_file))
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

    def _write_sacc_data(
        self, sacc_data: sacc.Sacc, output_path: Path, target_format: SaccFormat
    ) -> None:
        """Write SACC data object to output file."""
        self.console.print(f"Writing {target_format} file: [cyan]{output_path}[/cyan]")
        try:
            match target_format:
                case SaccFormat.FITS:
                    sacc_data.save_fits(str(output_path), overwrite=self.overwrite)
                case SaccFormat.HDF5:
                    if self.overwrite and output_path.exists():
                        output_path.unlink()
                    sacc_data.save_hdf5(str(output_path))
                case _ as unreachable:
                    assert_never(unreachable)
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
        input_size = self.sacc_file.stat().st_size
        output_size = output_path.stat().st_size

        self.console.print("\n" + "=" * 60)
        self.console.print("✅ [bold green]Transformation successful![/bold green]")
        self.console.print("=" * 60)
        input_info = f"({src_format.upper()}, {input_size:,} bytes)"
        self.console.print(f"Input:  [cyan]{self.sacc_file}[/cyan] {input_info}")
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

    def _fix_ordering(self, sacc_data: sacc.Sacc) -> None:
        """Fix tracer ordering issues in SACC data.

        This method detects and corrects tracer ordering violations where the order
        of tracers in a measurement doesn't match the order of measurement types in
        the SACC data type string.

        The SACC naming convention requires that tracer order matches the measurement
        type order in the data type string. For example:

        - Data type 'galaxy_shearDensity_cl_e' means (shear, density)
        - Tracers must be ordered as (shear_tracer, density_tracer)
        - If found as (density_tracer, shear_tracer), they will be swapped

        The method:

        1. Extracts metadata indices for all real and harmonic measurements
        2. Identifies measurements where tracer types are reversed (a > b)
        3. Swaps tracer order in data points to match canonical ordering
        4. Reports summary of corrections made

        Canonical ordering follows: CMB < Clusters < Galaxies

        Within galaxies: SHEAR < COUNTS

        :param sacc_data: SACC data object to fix (modified in-place)
        :type sacc_data: sacc.Sacc
        :returns: None (modifies sacc_data in place)
        :rtype: None

        .. seealso::
            :meth:`_report_ordering_corrections` - Reports summary of corrections made
        """
        real_indices = extract_all_real_metadata_indices(
            sacc_data, self.allow_mixed_types
        )
        harmonic_indices = extract_all_harmonic_metadata_indices(
            sacc_data, self.allow_mixed_types
        )
        tracers_to_fix = {}
        for real_index in real_indices:
            a, b = real_index["tracer_types"]
            if a > b:
                tracers_to_fix[
                    (real_index["data_type"], tuple(real_index["tracer_names"]))
                ] = real_index
        for harmonic_index in harmonic_indices:
            a, b = harmonic_index["tracer_types"]
            if a > b:
                tracers_to_fix[
                    (harmonic_index["data_type"], tuple(harmonic_index["tracer_names"]))
                ] = harmonic_index

        if not tracers_to_fix:
            self.console.print("No tracer ordering issues detected.")
            return

        self.console.print(
            f"[bold yellow]Fixing tracer ordering for {len(tracers_to_fix)} "
            f"unique corrections.[/bold yellow]"
        )

        # Track how many data points are corrected for each correction
        data_points = sacc_data.get_data_points()
        correction_counts = {key: 0 for key in tracers_to_fix}
        for dp in data_points:
            key = (dp.data_type, dp.tracers)
            if key in tracers_to_fix:
                assert len(dp.tracers) == 2
                dp.tracers = (dp.tracers[1], dp.tracers[0])
                correction_counts[key] += 1

        # Report corrections made
        self._report_ordering_corrections(correction_counts)

    def _report_ordering_corrections(
        self, correction_counts: dict[tuple[str, tuple[str, ...]], int]
    ) -> None:
        """Report summary of tracer ordering corrections made.

        This method prints a detailed summary of all tracer ordering corrections,
        showing for each correction the data type, affected tracers, and number of
        data points that were reordered.

        :param correction_counts: Dictionary mapping (data_type, tracers) tuples
            to the count of data points corrected for that combination
        :type correction_counts: dict[tuple[str, tuple[str, ...]], int]
        :returns: None
        :rtype: None

        .. rubric:: Example Output

        ::

            For data type galaxy_shearDensity_cl_e and tracers src0, lens0, 6 \
data points were flipped.
            For data type galaxy_shearDensity_cl_e and tracers src1, lens0, 7 \
data points were flipped.
        """
        for key, count in correction_counts.items():
            dt, tracers = key
            tracer_str = ", ".join(tracers)
            self.console.print(
                f"For data type [cyan]{dt}[/cyan] and tracers "
                f"[magenta]{tracer_str}[/magenta], [green]{count}[/green] "
                "data points were flipped."
            )
