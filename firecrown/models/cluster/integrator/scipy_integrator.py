"""The SciPy integrator module

This module holds the scipy implementation of the integrator classes
"""
from typing import Callable
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
        func_to_integrate: Callable,
    ) -> float:
        val = nquad(
            func_to_integrate,
            ranges=self.integral_bounds,
            args=self.extra_args,
            opts={
                "epsabs": self._absolute_tolerance,
                "epsrel": self._relative_tolerance,
            },
        )[0]
        assert isinstance(val, float)
        return val
