"""write me"""
from typing import Tuple
import numpy as np
import numpy.typing as npt
import pytest
from firecrown.models.cluster.integrator.scipy_integrator import (
    ScipyIntegrator,
)
from firecrown.models.cluster.kernel import KernelType, Kernel
from firecrown.models.cluster.abundance import ClusterAbundance


@pytest.fixture(name="cl_abundance")
def fixture_cl_abundance():
    cl_abundance = ClusterAbundance(
        min_z=0,
        max_z=2,
        min_mass=13,
        max_mass=17,
        sky_area=100,
        halo_mass_function=None,
    )
    return cl_abundance


class MockKernel(Kernel):
    """write me"""

    def distribution(
        self,
        _mass: npt.NDArray[np.float64],
        _z: npt.NDArray[np.float64],
        _mass_proxy: npt.NDArray[np.float64],
        _z_proxy: npt.NDArray[np.float64],
        _mass_proxy_limits: Tuple[float, float],
        _z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """The functional form of the distribution or spread of this kernel"""
        return np.atleast_1d(1.0)


def test_scipy_set_integration_bounds_no_kernels(cl_abundance: ClusterAbundance):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 2
        assert spi.integral_bounds == [(13, 17), (0, 2)]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_scipy_set_integration_bounds_dirac_delta(cl_abundance: ClusterAbundance):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    dd_kernel = MockKernel(KernelType.MASS_PROXY, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 2
        assert spi.integral_bounds == [mass_limits, (0, 2)]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }

    cl_abundance.kernels.clear()
    dd_kernel = MockKernel(KernelType.Z_PROXY, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 2
        assert spi.integral_bounds == [(13, 17), z_limits]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }
    dd_kernel2 = MockKernel(KernelType.MASS_PROXY, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 2
        assert spi.integral_bounds == [mass_limits, z_limits]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_scipy_set_integration_bounds_integrable_kernels(
    cl_abundance: ClusterAbundance,
):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    ig_kernel = MockKernel(KernelType.MASS_PROXY)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 3
        assert spi.integral_bounds == [(13, 17), (0, 2), mass_limits]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.MASS_PROXY: 2,
        }

    cl_abundance.kernels.clear()
    ig_kernel = MockKernel(KernelType.Z_PROXY)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 3
        assert spi.integral_bounds == [(13, 17), (0, 2), z_limits]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.Z_PROXY: 2,
        }

    ig_kernel2 = MockKernel(KernelType.MASS_PROXY)
    cl_abundance.add_kernel(ig_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 4
        assert spi.integral_bounds == [(13, 17), (0, 2), z_limits, mass_limits]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.Z_PROXY: 2,
            KernelType.MASS_PROXY: 3,
        }


def test_scipy_set_integration_bounds_analytic_slns(
    cl_abundance: ClusterAbundance,
):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    a_kernel = MockKernel(KernelType.MASS_PROXY, has_analytic_sln=True)
    cl_abundance.add_kernel(a_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 2
        assert spi.integral_bounds == [(13, 17), (0, 2)]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }

    a_kernel2 = MockKernel(KernelType.Z_PROXY, has_analytic_sln=True)
    cl_abundance.add_kernel(a_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        spi.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert spi.mass_proxy_limits == mass_limits
        assert spi.z_proxy_limits == z_limits
        assert len(spi.integral_bounds) == 2
        assert spi.integral_bounds == [(13, 17), (0, 2)]
        assert spi.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_scipy_integrator_integrate(cl_abundance: ClusterAbundance):
    spi = ScipyIntegrator()
    cl_abundance.min_mass = 0
    cl_abundance.max_mass = 1
    cl_abundance.min_z = 0
    cl_abundance.max_z = 1

    def integrand(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        _c: npt.NDArray[np.float64],
        _d: npt.NDArray[np.float64],
        _e: Tuple[float, float],
        _f: Tuple[float, float],
    ):
        # xy
        result = a * b
        return result

    spi.set_integration_bounds(
        cl_abundance,
        (0, 1),
        (0, 1),
    )
    result = spi.integrate(integrand)
    # \int_0^1 \int_0^1 xy dx dy = 1/4
    assert result == pytest.approx(0.25, rel=1e-15, abs=0)
