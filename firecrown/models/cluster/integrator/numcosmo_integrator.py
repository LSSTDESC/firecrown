"""The NumCosmo integrator module

This module holds the NumCosmo implementation of the integrator classes
"""
from typing import Tuple, Callable, Sequence, Optional, List
from enum import Enum
import numpy as np
import numpy.typing as npt
from numcosmo_py import Ncm


class NumCosmoIntegralMethod(Enum):
    """The available NumCosmo integration methods."""

    P = Ncm.IntegralNDMethod.P
    P_V = Ncm.IntegralNDMethod.P_V
    H = Ncm.IntegralNDMethod.H
    H_V = Ncm.IntegralNDMethod.H_V


class NumCosmoIntegrator:
    """The NumCosmo implementation of the Integrator base class."""

    def __init__(
        self,
        method: Optional[NumCosmoIntegralMethod] = None,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-12,
    ) -> None:
        super().__init__()
        self.method = method or NumCosmoIntegralMethod.P_V
        self.integral_bounds: List[Tuple[float, float]] = []
        self.extra_args: List[float] = []
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def integrate(self, func_to_integrate: Callable) -> float:
        Ncm.cfg_init()

        int_nd = CountsIntegralND(
            len(self.integral_bounds), func_to_integrate, self.extra_args
        )
        int_nd.set_method(self.method.value)
        int_nd.set_reltol(self._relative_tolerance)
        int_nd.set_abstol(self._absolute_tolerance)
        res = Ncm.Vector.new(1)
        err = Ncm.Vector.new(1)

        bl, bu = zip(*self.integral_bounds)
        int_nd.eval(Ncm.Vector.new_array(bl), Ncm.Vector.new_array(bu), res, err)
        return res.get(0)


class CountsIntegralND(Ncm.IntegralND):
    """Integral subclass used to compute the integrals using NumCosmo."""

    def __init__(
        self,
        dim: int,
        fun: Callable[[npt.NDArray], Sequence[float]],
        args: List[float],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fun = fun
        self.extra_args = args

    # pylint: disable-next=arguments-differ
    def do_get_dimensions(self) -> Tuple[int, int]:
        """Returns the dimensionality of the integral."""
        return self.dim, 1

    # pylint: disable-next=arguments-differ
    def do_integrand(
        self,
        x_vec: Ncm.Vector,
        dim: int,
        npoints: int,
        _fdim: int,
        fval_vec: Ncm.Vector,
    ) -> None:
        """Called by NumCosmo to evaluate the integrand."""
        x = np.array(x_vec.dup_array()).reshape(npoints, dim)
        fval_vec.set_array(self.fun(x, *self.extra_args))
