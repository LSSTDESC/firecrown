from numcosmo_py import Ncm
from typing import Tuple
from abc import ABC, abstractmethod
from scipy.integrate import nquad
import numpy as np


class Integrator(ABC):
    @abstractmethod
    def integrate(self, integrand, bounds, bounds_map, extra_args):
        pass


class ScipyIntegrator(Integrator):
    def __init__(self, relative_tolerance=1e-4, absolute_tolerance=1e-12):
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def integrate(self, integrand, bounds, bounds_map, extra_args):
        cc = nquad(
            integrand,
            ranges=bounds,
            args=(extra_args, bounds_map),
            opts={
                "epsabs": self._absolute_tolerance,
                "epsrel": self._relative_tolerance,
            },
        )[0]

        return cc


class NumCosmoIntegrator(Integrator):
    def __init__(self, relative_tolerance=1e-4, absolute_tolerance=1e-12):
        super().__init__()
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def integrate(self, integrand, bounds, bounds_map, extra_args):
        super().__init__()
        Ncm.cfg_init()
        int_nd = CountsIntegralND(
            len(bounds),
            integrand,
            extra_args,
            bounds_map,
        )
        int_nd.set_method(Ncm.IntegralNDMethod.P_V)
        int_nd.set_reltol(self._relative_tolerance)
        int_nd.set_abstol(self._absolute_tolerance)
        res = Ncm.Vector.new(1)
        err = Ncm.Vector.new(1)

        bl, bu = zip(*bounds)
        int_nd.eval(Ncm.Vector.new_array(bl), Ncm.Vector.new_array(bu), res, err)
        return res.get(0)


class CountsIntegralND(Ncm.IntegralND):
    """Integral subclass used by the ClusterAbundance
    to compute the integrals using numcosmo."""

    def __init__(self, dim, fun, *args):
        super().__init__()
        self.dim = dim
        self.fun = fun
        self.args = args

    # pylint: disable-next=arguments-differ
    def do_get_dimensions(self) -> Tuple[int, int]:
        """Get number of dimensions."""
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
        """Integrand function."""
        x = np.array(x_vec.dup_array()).reshape(npoints, dim)
        fval_vec.set_array([self.fun(x_i, *self.args) for x_i in x])
