"""This module deals with two-point functions metadata.

It contains all data classes and functions for store and extract two-point functions
metadata from a sacc file.
"""

from typing import Optional
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import sacc


@dataclass(frozen=True)
class TracerNames:
    """The names of the two tracers in the sacc file."""

    name1: str
    name2: str

    def __getitem__(self, item):
        """Get the name of the tracer at the given index."""
        if item == 0:
            return self.name1
        if item == 1:
            return self.name2
        raise IndexError

    def __iter__(self):
        """Iterate through the data members.

        This is to allow automatic unpacking.
        """
        yield self.name1
        yield self.name2


TRACER_NAMES_TOTAL = TracerNames("", "")  # special name to represent total


# kw_only=True only available in Python >= 3.10:
# TODO update when we drop Python 3.9
@dataclass()
class Window:
    """The class used to represent a window function.

    It contains the ells at which the window function is defined, the weights
    of the window function, and the ells at which the window function is
    interpolated.

    It may contain the ells for interpolation if the theory prediction is
    calculated at a different set of ells than the window function.
    """

    ells: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]
    ells_for_interpolation: Optional[npt.NDArray[np.int64]] = None

    def __post_init__(self) -> None:
        """Make sure the weights have the right shape."""
        if len(self.weights.shape) != 2:
            raise ValueError("Weights should be a 2D array.")
        if self.weights.shape[0] != len(self.ells):
            raise ValueError("Weights should have the same number of rows as ells.")

    def n_observations(self) -> int:
        """Return the number of observations supported by the window function."""
        return self.weights.shape[1]


def extract_window_function(
    sacc_data: sacc.Sacc, indices: npt.NDArray[np.int64]
) -> Optional[Window]:
    """Extract a window function from a sacc file that matches the given indices.

    If there is no appropriate window function, return None.
    """
    bandpower_window = sacc_data.get_bandpower_windows(indices)
    if bandpower_window is None:
        return None
    return Window(
        ells=bandpower_window.values,
        weights=bandpower_window.weight / bandpower_window.weight.sum(axis=0),
    )
