"""Checked integer conversion for exact multipoles and SACC indices."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt

_IntBit = TypeVar("_IntBit", bound=npt.NBitBase)


def _as_int64(values: npt.NDArray[np.generic]) -> npt.NDArray[np.int64]:
    """Retain integer meaning and reject masks or overflow before conversion."""
    if np.ma.isMaskedArray(values):
        raise TypeError("Integer arrays must not be masked.")
    if values.dtype.kind not in "iu":
        raise TypeError("Exact multipoles and indices must have an integer dtype.")
    limits = np.iinfo(np.int64)
    if values.size and (
        int(values.min()) < limits.min or int(values.max()) > limits.max
    ):
        raise ValueError("Integer values are outside the int64 range.")
    return np.asarray(values, dtype=np.int64)


def _as_multipole_array(
    values: npt.NDArray[np.generic],
) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """Preserve exact integer support or real harmonic sampling coordinates."""
    if np.ma.isMaskedArray(values):
        raise TypeError("Multipole arrays must not be masked.")
    if values.dtype.kind in "iu":
        return _as_int64(values)
    if values.dtype.kind == "f":
        result = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("Sampling multipoles must be finite.")
        return result
    raise TypeError("Multipoles must have an integer or floating dtype.")
