"""SACC-related operations for Firecrown."""

from collections.abc import Generator

import numpy as np
import sacc
from numpy import typing as npt


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
