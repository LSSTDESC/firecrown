"""Tests for the numcosmo integrator module."""
from typing import Tuple
from unittest.mock import Mock
import numpy as np
import numpy.typing as npt
import pytest
from firecrown.models.cluster.integrator.integrator import Integrator
from firecrown.models.cluster.integrator.scipy_integrator import ScipyIntegrator
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.kernel import KernelType, Kernel
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand


@pytest.fixture(name="integrator", params=[ScipyIntegrator, NumCosmoIntegrator])
def fixture_integrator(request) -> Integrator:
    return request.param()


def test_numcosmo_set_integration_bounds_no_kernels(
    empty_cluster_abundance: ClusterAbundance, integrator: Integrator
):
    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))
    sky_area = 100**2

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 2
        assert integrator.integral_bounds == [(13, 17), (0, 2)]
        assert integrator.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_numcosmo_set_integration_bounds_dirac_delta(
    empty_cluster_abundance: ClusterAbundance, integrator: Integrator
):
    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))
    sky_area = 100**2

    dd_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        is_dirac_delta=True,
        has_analytic_sln=False,
    )
    dd_kernel.distribution.return_value = np.atleast_1d(1.0)
    empty_cluster_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 2
        assert integrator.integral_bounds == [mass_limits, (0, 2)]
        assert integrator.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }

    empty_cluster_abundance.kernels.clear()
    dd_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.Z_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=True,
    )
    dd_kernel.distribution.return_value = np.atleast_1d(1.0)
    empty_cluster_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 2
        assert integrator.integral_bounds == [(13, 17), z_limits]
        assert integrator.integral_args_lkp == {
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
    empty_cluster_abundance.add_kernel(dd_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 2
        assert integrator.integral_bounds == [mass_limits, z_limits]
        assert integrator.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_numcosmo_set_integration_bounds_integrable_kernels(
    empty_cluster_abundance: ClusterAbundance, integrator: Integrator
):
    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))
    sky_area = 100**2

    ig_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=False,
    )
    ig_kernel.distribution.return_value = np.atleast_1d(1.0)
    empty_cluster_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 3
        assert integrator.integral_bounds == [(13, 17), (0, 2), mass_limits]
        assert integrator.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.MASS_PROXY: 2,
        }

    empty_cluster_abundance.kernels.clear()
    ig_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.Z_PROXY,
        has_analytic_sln=False,
        is_dirac_delta=False,
    )
    ig_kernel.distribution.return_value = np.atleast_1d(1.0)
    empty_cluster_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 3
        assert integrator.integral_bounds == [(13, 17), (0, 2), z_limits]
        assert integrator.integral_args_lkp == {
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
    empty_cluster_abundance.add_kernel(ig_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 4
        assert integrator.integral_bounds == [(13, 17), (0, 2), z_limits, mass_limits]
        assert integrator.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
            KernelType.Z_PROXY: 2,
            KernelType.MASS_PROXY: 3,
        }


def test_numcosmo_set_integration_bounds_analytic_slns(
    empty_cluster_abundance: ClusterAbundance, integrator: Integrator
):
    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))
    sky_area = 100**2

    a_kernel = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        has_analytic_sln=True,
        is_dirac_delta=False,
    )
    a_kernel.distribution.return_value = np.atleast_1d(1.0)
    empty_cluster_abundance.add_kernel(a_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 2
        assert integrator.integral_bounds == [(13, 17), (0, 2)]
        assert integrator.integral_args_lkp == {
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
    empty_cluster_abundance.add_kernel(a_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        integrator.set_integration_bounds(
            empty_cluster_abundance, sky_area, z_limits, mass_limits
        )

        assert integrator.mass_proxy_limits == mass_limits
        assert integrator.z_proxy_limits == z_limits
        assert len(integrator.integral_bounds) == 2
        assert integrator.integral_bounds == [(13, 17), (0, 2)]
        assert integrator.integral_args_lkp == {
            KernelType.MASS: 0,
            KernelType.Z: 1,
        }


def test_numcosmo_integrator_integrate(
    empty_cluster_abundance: ClusterAbundance, integrator: Integrator
):
    empty_cluster_abundance.min_mass = 0
    empty_cluster_abundance.max_mass = 1
    empty_cluster_abundance.min_z = 0
    empty_cluster_abundance.max_z = 1
    sky_area = 100**2

    def integrand(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        _c: float,
        _d: npt.NDArray[np.float64],
        _e: npt.NDArray[np.float64],
        _f: Tuple[float, float],
        _g: Tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        # xy
        result = a * b
        return result

    integrator.set_integration_bounds(
        empty_cluster_abundance,
        sky_area,
        (0, 1),
        (0, 1),
    )
    result = integrator.integrate(integrand)
    # \int_0^1 \int_0^1 xy dx dy = 1/4
    assert result == pytest.approx(0.25, rel=1e-15, abs=0)
