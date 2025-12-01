"""Utility functions for data types.

This module contains utility functions for working with data and theory vectors.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from firecrown.data_types._vectors import DataVector, TheoryVector


def residuals(data: DataVector, theory: TheoryVector) -> npt.NDArray[np.float64]:
    """Return a bare np.ndarray with the difference between `data` and `theory`.

    This is to be preferred to using arithmetic on the vectors directly.
    """
    assert isinstance(data, DataVector)
    assert isinstance(theory, TheoryVector)
    return (data - theory).view(np.ndarray)
