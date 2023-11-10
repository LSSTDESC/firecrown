"""Tests for the numcosmo integrator module."""
from typing import Tuple
import numpy as np
import numpy.typing as npt
import pytest
from unittest.mock import Mock
from firecrown.models.cluster.integrator.numcosmo_integrator import (
    NumCosmoIntegrator,
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


def test_numcosmo_set_integration_bounds_no_kernels(cl_abundance: ClusterAbundance):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [(13, 17), (0, 2)]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_numcosmo_set_integration_bounds_dirac_delta(cl_abundance: ClusterAbundance):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    dd_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        is_dirac_delta=True,
        has_analytic_sln=False,
    )
    dd_kernel.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [mass_limits, (0, 2)]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }

    cl_abundance.kernels.clear()
    dd_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.Z_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=True,
    )
    dd_kernel.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [(13, 17), z_limits]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }
    dd_kernel2 = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=True,
    )
    dd_kernel2.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(dd_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [mass_limits, z_limits]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_numcosmo_set_integration_bounds_integrable_kernels(
    cl_abundance: ClusterAbundance,
):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    ig_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=False,
    )
    ig_kernel.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 3
        assert nci.integral_bounds == [(13, 17), (0, 2), mass_limits]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.MASS_PROXY: 2,
        }

    cl_abundance.kernels.clear()
    ig_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.Z_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=False,
    )
    ig_kernel.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 3
        assert nci.integral_bounds == [(13, 17), (0, 2), z_limits]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.Z_PROXY: 2,
        }

    ig_kernel2 = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=False,
    )
    ig_kernel2.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(ig_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 4
        assert nci.integral_bounds == [(13, 17), (0, 2), z_limits, mass_limits]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.Z_PROXY: 2,
            KernelType.MASS_PROXY: 3,
        }


def test_numcosmo_set_integration_bounds_analytic_slns(
    cl_abundance: ClusterAbundance,
):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    a_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        has_analytic_sln=True,
        is_dirac_delta=False,
    )
    a_kernel.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(a_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [(13, 17), (0, 2)]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }
    a_kernel2 = Mock(
        spec=Kernel,
        kernel_type=KernelType.Z_PROXY,
        has_analytic_sln=True,
        is_dirac_delta=False,
    )
    a_kernel2.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(a_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [(13, 17), (0, 2)]
        assert nci.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_numcosmo_integrator_integrate(cl_abundance: ClusterAbundance):
    nci = NumCosmoIntegrator()
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

    nci.set_integration_bounds(
        cl_abundance,
        (0, 1),
        (0, 1),
    )
    result = nci.integrate(integrand)
    # \int_0^1 \int_0^1 xy dx dy = 1/4
    assert result == 0.25
