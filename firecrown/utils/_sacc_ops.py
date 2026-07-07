"""SACC-related operations for Firecrown."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import sacc
from numpy import typing as npt
from sacc.utils import detect_sacc_file_type
from typing_extensions import assert_never


def load_sacc_data(filepath: str | Path) -> sacc.Sacc:
    """Load SACC data from a file, auto-detecting the format.

    Uses SACC's own format detection to determine whether the file is in
    HDF5 or FITS format, then dispatches to the appropriate loader. This
    allows the function to work with both modern HDF5-based SACC files and
    legacy FITS-based SACC files, including gzip-compressed variants, and
    regardless of the file's extension.

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

    try:
        file_type = detect_sacc_file_type(str(file_path))
    except ValueError as exc:
        raise ValueError(
            f"Failed to load SACC data from file: {file_path}\n"
            f"The file could not be read as either HDF5 or FITS format.\n"
            f"{exc}"
        ) from exc

    if file_type == "hdf5":
        return sacc.Sacc.load_hdf5(str(file_path))
    return sacc.Sacc.load_fits(str(file_path))


def ensure_path(file: str | Path) -> Path:
    """Ensure the file path is a Path object."""
    match file:
        case str():
            return Path(file)
        case Path():
            return file
        case _ as unreachable:
            assert_never(unreachable)


def upper_triangle_indices(n: int) -> Generator[tuple[int, int], None, None]:
    """Returns the upper triangular indices for an (n x n) matrix.

    generator that yields a sequence of tuples that carry the indices for an
    (n x n) upper-triangular matrix. This is a replacement for the nested loops:

    for i in range(n):
      for j in range(i, n):
        ...

    :param n: the size of the matrix
    :return: the generator
    """
    for i in range(n):
        for j in range(i, n):
            yield i, j


def save_to_sacc(
    sacc_data: sacc.Sacc,
    data_vector: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    strict: bool = True,
) -> sacc.Sacc:
    """Save a data vector into a (new) SACC object, copied from `sacc_data`.

    Note that the original object `sacc_data` is not modified. Its contents are
    copied into a new object, and the new information is put into that copy,
    which is returned by this method.

    If `strict` is True (the default), then we must overwrite the entire data
    vector. If `strict` is False, then we only overwrite the data at the
    specified indices.

    :param sacc_data: SACC object to be copied. It is not modified.
    :param data_vector: Data vector to be saved to the new copy of `sacc_data`.
    :param indices: SACC indices where the data vector should be written.
    :param strict: Whether to check if the data vector covers all the data
        already present in the sacc_data.
    :return: A copy of `sacc_data`, with data at `indices` replaced with `data_vector`.
    """
    assert len(indices) == len(data_vector)

    new_sacc = sacc_data.copy()

    if strict:
        if set(indices.ravel().tolist()) != set(sacc_data.indices()):
            raise RuntimeError(
                "The data to be saved does not cover all the data in the "
                "sacc object. To write only the calculated predictions, "
                "set strict=False."
            )

    for data_idx, sacc_idx in enumerate(indices):
        new_sacc.data[sacc_idx].value = data_vector[data_idx]

    return new_sacc
