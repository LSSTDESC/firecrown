"""The cluster integrator module.

This module holds the classes that define the interface required to
integrate a function, along with specific implementations using SciPy and NumCosmo.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from numcosmo_py import Ncm
from scipy.integrate import nquad


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


class NumCosmoIntegralMethod(Enum):
    """The available NumCosmo integration methods."""

    P = Ncm.IntegralNDMethod.P
    P_V = Ncm.IntegralNDMethod.P_V
    H = Ncm.IntegralNDMethod.H
    H_V = Ncm.IntegralNDMethod.H_V


class NumCosmoIntegrator(Integrator):
    """The NumCosmo implementation of the Integrator base class."""

    def __init__(
        self,
        method: None | NumCosmoIntegralMethod = None,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-12,
    ) -> None:
        super().__init__()
        self.method = method or NumCosmoIntegralMethod.P_V
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def integrate(
        self,
        func_to_integrate: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
    ) -> float:
        """Integrate the provided integrand argument with NumCosmo."""
        Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

        int_nd = CountsIntegralND(
            len(self.integral_bounds), func_to_integrate, self.extra_args
        )
        int_nd.set_method(self.method.value)
        int_nd.set_reltol(self._relative_tolerance)
        int_nd.set_abstol(self._absolute_tolerance)
        res = Ncm.Vector.new(1)
        err = Ncm.Vector.new(1)

        bl, bu = zip(*self.integral_bounds, strict=False)
        int_nd.eval(Ncm.Vector.new_array(bl), Ncm.Vector.new_array(bu), res, err)
        return res.get(0)  # pylint: disable-msg=no-member


class CountsIntegralND(Ncm.IntegralND):
    """Integral subclass used to compute the integrals using NumCosmo."""

    def __init__(
        self,
        dim: int,
        fun: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        args: npt.NDArray[np.float64],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fun = fun
        self.extra_args = args

    # pylint: disable-next=arguments-differ
    def do_get_dimensions(self) -> tuple[int, int]:
        """Returns the dimensionality of the integral."""
        return self.dim, 1

    # pylint: disable-next=arguments-differ
    def do_integrand(
        self,
        x: Ncm.Vector,
        dim: int,
        npoints: int,
        _fdim: int,
        fval: Ncm.Vector,
    ) -> None:
        """Called by NumCosmo to evaluate the integrand."""
        x_array = np.array(x.dup_array()).reshape(npoints, dim)
        fval.set_array(list(self.fun(x_array, self.extra_args)))
