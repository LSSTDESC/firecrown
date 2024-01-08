"""Tests for the integrator module."""

import numpy as np
import numpy.typing as npt
import pytest
from firecrown.models.cluster.integrator.scipy_integrator import ScipyIntegrator
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator


# @pytest.fixture(name="integrator", params=[ScipyIntegrator, NumCosmoIntegrator])
# def fixture_integrator(request) -> Integrator:
#     return request.param()


def test_numcosmo_integrator_integrate():
    integrator = NumCosmoIntegrator()

    def integrand(int_args: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # xy
        a = int_args[:, 0]
        b = int_args[:, 1]
        result = a * b
        return result

    integrator.integral_bounds = [(0, 1), (0, 1)]
    integrator.extra_args = []
    result = integrator.integrate(integrand)
    # \int_0^1 \int_0^1 xy dx dy = 1/4
    assert result == pytest.approx(0.25, rel=1e-15, abs=0)


def test_scipy_integrator_integrate():
    integrator = ScipyIntegrator()

    def integrand(a: np.float64, b: np.float64) -> np.float64:
        # xy
        result = a * b
        return result

    integrator.integral_bounds = [(0, 1), (0, 1)]
    integrator.extra_args = []
    result = integrator.integrate(integrand)
    # \int_0^1 \int_0^1 xy dx dy = 1/4
    assert result == pytest.approx(0.25, rel=1e-15, abs=0)
