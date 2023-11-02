import numpy as np
import numpy.typing as npt
import pytest
from typing import List, Tuple, Optional
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


class MockKernel(Kernel):
    def __init__(
        self,
        kernel_type: KernelType,
        is_dirac_delta: bool = False,
        has_analytic_sln: bool = False,
        integral_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        super().__init__(kernel_type, is_dirac_delta, has_analytic_sln, integral_bounds)

    def distribution(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        z_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Tuple[float, float],
        z_proxy_limits: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """The functional form of the distribution or spread of this kernel"""
        return np.atleast_1d(1.0)


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
            KernelType.mass: 0,
            KernelType.z: 1,
        }


def test_numcosmo_set_integration_bounds_dirac_delta(cl_abundance: ClusterAbundance):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    dd_kernel = MockKernel(KernelType.mass_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [mass_limits, (0, 2)]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
        }

    cl_abundance.kernels.clear()
    dd_kernel = MockKernel(KernelType.z_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [(13, 17), z_limits]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
        }
    dd_kernel2 = MockKernel(KernelType.mass_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [mass_limits, z_limits]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
        }


def test_numcosmo_set_integration_bounds_integrable_kernels(
    cl_abundance: ClusterAbundance,
):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    ig_kernel = MockKernel(KernelType.mass_proxy)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 3
        assert nci.integral_bounds == [(13, 17), (0, 2), mass_limits]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
            KernelType.mass_proxy: 2,
        }

    cl_abundance.kernels.clear()
    ig_kernel = MockKernel(KernelType.z_proxy)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 3
        assert nci.integral_bounds == [(13, 17), (0, 2), z_limits]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
            KernelType.z_proxy: 2,
        }

    ig_kernel2 = MockKernel(KernelType.mass_proxy)
    cl_abundance.add_kernel(ig_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 4
        assert nci.integral_bounds == [(13, 17), (0, 2), z_limits, mass_limits]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
            KernelType.z_proxy: 2,
            KernelType.mass_proxy: 3,
        }


def test_numcosmo_set_integration_bounds_analytic_slns(
    cl_abundance: ClusterAbundance,
):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    a_kernel = MockKernel(KernelType.mass_proxy, has_analytic_sln=True)
    cl_abundance.add_kernel(a_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [(13, 17), (0, 2)]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
        }

    a_kernel2 = MockKernel(KernelType.z_proxy, has_analytic_sln=True)
    cl_abundance.add_kernel(a_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        nci.set_integration_bounds(cl_abundance, z_limits, mass_limits)

        assert nci.mass_proxy_limits == mass_limits
        assert nci.z_proxy_limits == z_limits
        assert len(nci.integral_bounds) == 2
        assert nci.integral_bounds == [(13, 17), (0, 2)]
        assert nci.integral_args_lkp == {
            KernelType.mass: 0,
            KernelType.z: 1,
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
        c: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64],
        e: Tuple[float, float],
        f: Tuple[float, float],
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
