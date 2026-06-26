"""Utilities for loading and handling SACC data files."""

from pathlib import Path

import sacc
from typing_extensions import assert_never


def load_sacc_data(filepath: str | Path) -> sacc.Sacc:
    """Load SACC data from a file, auto-detecting the format.

    Attempts to load the file first as HDF5, then as FITS if HDF5 fails.
    This allows the function to work with both modern HDF5-based SACC files
    and legacy FITS-based SACC files.

    :param filepath: Path to the SACC data file (str or Path object)
    :return: Loaded SACC data object
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the file cannot be read as either HDF5 or FITS SACC data
    """
    # Convert to Path object for consistent handling
    file_path = Path(filepath) if isinstance(filepath, str) else filepath

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"SACC file not found: {file_path}")

    # Try HDF5 first (modern format)
    hdf5_error = None
    try:
        return sacc.Sacc.load_hdf5(str(file_path))
    except OSError as e:
        hdf5_error = e

    # If HDF5 failed, try FITS (legacy format)
    fits_error = None
    try:
        return sacc.Sacc.load_fits(str(file_path))
    except OSError as e:
        fits_error = e

    # Both formats failed - provide helpful error message
    raise ValueError(
        f"Failed to load SACC data from file: {file_path}\n"
        f"The file could not be read as either HDF5 or FITS format.\n"
        f"HDF5 error: {hdf5_error}\n"
        f"FITS error: {fits_error}"
    )


def ensure_path(file: str | Path) -> Path:
    """Ensure the file path is a Path object."""
    match file:
        case str():
            return Path(file)
        case Path():
            return file
        case _ as unreachable:
            assert_never(unreachable)
