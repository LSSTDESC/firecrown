"""The cluster integrator module.

This module holds the classes that define the interface required to
integrate a function.
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import numpy.typing as npt


class Integrator(ABC):
    """The integrator base class.

    This class acts as an adapter around an integration library.
    """

    def __init__(self) -> None:
        self.integral_bounds: list[tuple[float, float]] = []
        self.extra_args: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

    @abstractmethod
    def integrate(
        self,
        func_to_integrate: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
    ) -> float:
        """Call this method to integrate the provided integrand argument."""
