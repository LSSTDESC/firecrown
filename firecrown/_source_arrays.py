"""Real-array ingress for tomography and galaxy source preparation."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt

_NBit = TypeVar("_NBit", bound=npt.NBitBase)
_DndzBit = TypeVar("_DndzBit", bound=npt.NBitBase)
_RealArray = npt.NDArray[np.integer[_NBit] | np.floating[_NBit]]


def _as_float64(values: _RealArray[_NBit]) -> npt.NDArray[np.float64]:
    """Reject unsupported representations before converting physical values."""
    if np.ma.isMaskedArray(values):
        raise TypeError("Source arrays must not be masked.")
    if values.dtype.kind not in "iuf":
        raise TypeError("Source arrays must have a real integer or floating dtype.")
    return np.asarray(values, dtype=np.float64)


def _as_float64_pair(
    values: tuple[_RealArray[_NBit], _RealArray[_NBit]] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Convert optional tabulated source-systematic coordinates and values."""
    if values is None:
        return None
    return _as_float64(values[0]), _as_float64(values[1])
