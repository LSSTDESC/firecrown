"""Utilities for loading and handling SACC data files."""

from pathlib import Path

import sacc
from typing_extensions import assert_never


def load_sacc_data(filepath: str | Path) -> sacc.Sacc:
    """Load SACC data from a file, auto-detecting the format.

    Delegates to sacc.Sacc.load(), which auto-detects FITS vs HDF5 based on
    filename extension or (for ambiguous extensions like .sacc) file content.

    :param filepath: Path to the SACC data file (str or Path object)
    :return: Loaded SACC data object
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the file cannot be recognized as SACC data
    """
    # Convert to Path object for consistent handling
    file_path = Path(filepath) if isinstance(filepath, str) else filepath

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"SACC file not found: {file_path}")

    try:
        return sacc.Sacc.load(str(file_path))
    except ValueError as e:
        raise ValueError(f"Failed to load SACC data from file: {file_path}\n{e}") from e


def ensure_path(file: str | Path) -> Path:
    """Ensure the file path is a Path object."""
    match file:
        case str():
            return Path(file)
        case Path():
            return file
        case _ as unreachable:
            assert_never(unreachable)
