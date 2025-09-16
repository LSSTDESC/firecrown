"""The SciPy integrator module.

This module holds the scipy implementation of the integrator classes
"""

from typing import Callable, Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import nquad

from firecrown.models.cluster.integrator.integrator import Integrator


class ScipyIntegrator(Integrator):
    """The scipy implementation of the Integrator base class using nquad."""

    def __init__(
        self, relative_tolerance: float = 1e-4, absolute_tolerance: float = 1e-12
    ) -> None:
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def integrate(
        self,
        func_to_integrate: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
    ) -> float:
        """Integrate the provided integrand argument with SciPy."""
        arg_len = len(self.integral_bounds)

        def wrapper(*args: Any) -> float:
            a = np.array(args[:arg_len])
            b = np.array(args[arg_len:])
            result = func_to_integrate(a, b)
            assert isinstance(result, np.ndarray)
            assert len(result) == 1
            return result[0]

        val = nquad(
            wrapper,
            ranges=self.integral_bounds,
            args=self.extra_args,
            opts={
                "epsabs": self._absolute_tolerance,
                "epsrel": self._relative_tolerance,
            },
        )[0]
        assert isinstance(val, float)
        return val
