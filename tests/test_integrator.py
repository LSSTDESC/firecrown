import numpy as np
import pytest
from typing import List, Tuple
from firecrown.integrator.numcosmo_integrator import (
    NumCosmoArgReader,
    NumCosmoIntegrator,
)
from firecrown.integrator.scipy_integrator import ScipyArgReader, ScipyIntegrator
from firecrown.models.cluster.kernel import KernelType, Kernel, ArgReader
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
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        super().__init__(kernel_type, is_dirac_delta, has_analytic_sln, integral_bounds)

    def distribution(self, args: List[float], arg_reader: ArgReader):
        return 1.0


def test_numcosmo_argreader_extra_args():
    arg_reader = NumCosmoArgReader()

    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    arg_reader.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}

    extra_args = [(1, 2, 3), ("hello world")]
    arg_reader.extra_args = {KernelType.mass_proxy.name: 0, KernelType.z_proxy.name: 1}

    integral_bounds = np.array(list(zip(mass, z)))
    int_args = [integral_bounds, extra_args]

    assert arg_reader.get_extra_args(int_args, KernelType.mass_proxy) == (1, 2, 3)
    assert arg_reader.get_extra_args(int_args, KernelType.z_proxy) == "hello world"


def test_numcosmo_argreader_integral_bounds():
    arg_reader = NumCosmoArgReader()

    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    z_proxy = np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01])
    mass_proxy = np.array([0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9])

    arg_reader.integral_bounds = {
        KernelType.mass.name: 0,
        KernelType.z.name: 1,
        KernelType.z_proxy.name: 2,
        KernelType.mass_proxy.name: 3,
    }

    integral_bounds = np.array(list(zip(mass, z, z_proxy, mass_proxy)))
    int_args = [integral_bounds]

    assert (mass == arg_reader.get_integral_bounds(int_args, KernelType.mass)).all()
    assert (z == arg_reader.get_integral_bounds(int_args, KernelType.z)).all()
    assert (
        z_proxy == arg_reader.get_integral_bounds(int_args, KernelType.z_proxy)
    ).all()
    assert (
        mass_proxy == arg_reader.get_integral_bounds(int_args, KernelType.mass_proxy)
    ).all()


def test_create_numcosmo_argreader():
    am = NumCosmoArgReader()
    assert am.integral_bounds == dict()
    assert am.extra_args == dict()
    assert am.integral_bounds_idx == 0
    assert am.extra_args_idx == 1


def test_scipy_argreader_extra_args():
    arg_reader = ScipyArgReader()

    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    arg_reader.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}

    extra_args = [(1, 2, 3), ("hello world")]
    arg_reader.extra_args = {KernelType.mass_proxy.name: 2, KernelType.z_proxy.name: 3}

    for m_i, z_i in list(zip(mass, z)):
        int_args = [m_i, z_i, *extra_args]

        assert arg_reader.get_extra_args(int_args, KernelType.mass_proxy) == (1, 2, 3)
        assert arg_reader.get_extra_args(int_args, KernelType.z_proxy) == "hello world"


def test_scipy_argreader_integral_bounds():
    arg_reader = ScipyArgReader()

    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    z_proxy = np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01])
    mass_proxy = np.array([0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9])

    arg_reader.integral_bounds = {
        KernelType.mass.name: 0,
        KernelType.z.name: 1,
        KernelType.z_proxy.name: 2,
        KernelType.mass_proxy.name: 3,
    }

    for m_i, z_i, zp_i, mp_i in list(zip(mass, z, z_proxy, mass_proxy)):
        int_args = [m_i, z_i, zp_i, mp_i]
        assert m_i == arg_reader.get_integral_bounds(int_args, KernelType.mass)
        assert z_i == arg_reader.get_integral_bounds(int_args, KernelType.z)
        assert zp_i == arg_reader.get_integral_bounds(int_args, KernelType.z_proxy)
        assert mp_i == arg_reader.get_integral_bounds(int_args, KernelType.mass_proxy)


def test_create_scipy_argreader():
    am = ScipyArgReader()
    assert am.integral_bounds == dict()
    assert am.extra_args == dict()
    assert am.integral_bounds_idx == 0


def test_numcosmo_get_integration_bounds_no_kernels(cl_abundance: ClusterAbundance):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }


def test_numcosmo_get_integration_bounds_dirac_delta(cl_abundance: ClusterAbundance):
    nci = NumCosmoIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    dd_kernel = MockKernel(KernelType.mass_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [mass_limits, (0, 2)]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    cl_abundance.kernels.clear()
    dd_kernel = MockKernel(KernelType.z_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [(13, 17), z_limits]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    dd_kernel2 = MockKernel(KernelType.mass_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [mass_limits, z_limits]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }


def test_numcosmo_get_integration_bounds_integrable_kernels(
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
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 3
        assert bounds == [(13, 17), (0, 2), mass_limits]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.mass_proxy.name: 2,
        }

    cl_abundance.kernels.clear()
    ig_kernel = MockKernel(KernelType.z_proxy)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 3
        assert bounds == [(13, 17), (0, 2), z_limits]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.z_proxy.name: 2,
        }

    ig_kernel2 = MockKernel(KernelType.mass_proxy)
    cl_abundance.add_kernel(ig_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 4
        assert bounds == [(13, 17), (0, 2), z_limits, mass_limits]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.z_proxy.name: 2,
            KernelType.mass_proxy.name: 3,
        }

    p_kernel = MockKernel(KernelType.purity, integral_bounds=(0, 1))
    cl_abundance.add_kernel(p_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 5
        assert bounds == [(13, 17), (0, 2), z_limits, mass_limits, (0, 1)]
        assert len(nci.arg_reader.extra_args) == 0
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.z_proxy.name: 2,
            KernelType.mass_proxy.name: 3,
            KernelType.purity.name: 4,
        }


def test_numcosmo_get_integration_bounds_analytic_slns(
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
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 1
        assert extra_args == [mass_limits]
        assert nci.arg_reader.extra_args == {KernelType.mass_proxy.name: 0}

        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    a_kernel2 = MockKernel(KernelType.z_proxy, has_analytic_sln=True)
    cl_abundance.add_kernel(a_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 2
        assert extra_args == [mass_limits, z_limits]
        assert nci.arg_reader.extra_args == {
            KernelType.mass_proxy.name: 0,
            KernelType.z_proxy.name: 1,
        }

        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    a_kernel3 = MockKernel(
        KernelType.purity, has_analytic_sln=True, integral_bounds=(0, 1)
    )
    cl_abundance.add_kernel(a_kernel3)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = nci.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 3
        assert extra_args == [mass_limits, z_limits, (0, 1)]
        assert nci.arg_reader.extra_args == {
            KernelType.mass_proxy.name: 0,
            KernelType.z_proxy.name: 1,
            KernelType.purity.name: 2,
        }

        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert nci.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }


def test_scipy_get_integration_bounds_no_kernels(cl_abundance: ClusterAbundance):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }


def test_scipy_get_integration_bounds_dirac_delta(cl_abundance: ClusterAbundance):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    dd_kernel = MockKernel(KernelType.mass_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [mass_limits, (0, 2)]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    cl_abundance.kernels.clear()
    dd_kernel = MockKernel(KernelType.z_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [(13, 17), z_limits]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    dd_kernel2 = MockKernel(KernelType.mass_proxy, is_dirac_delta=True)
    cl_abundance.add_kernel(dd_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 2
        assert bounds == [mass_limits, z_limits]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }


def test_scipy_get_integration_bounds_integrable_kernels(
    cl_abundance: ClusterAbundance,
):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    ig_kernel = MockKernel(KernelType.mass_proxy)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 3
        assert bounds == [(13, 17), (0, 2), mass_limits]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.mass_proxy.name: 2,
        }

    cl_abundance.kernels.clear()
    ig_kernel = MockKernel(KernelType.z_proxy)
    cl_abundance.add_kernel(ig_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 3
        assert bounds == [(13, 17), (0, 2), z_limits]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.z_proxy.name: 2,
        }

    ig_kernel2 = MockKernel(KernelType.mass_proxy)
    cl_abundance.add_kernel(ig_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 4
        assert bounds == [(13, 17), (0, 2), z_limits, mass_limits]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.z_proxy.name: 2,
            KernelType.mass_proxy.name: 3,
        }

    p_kernel = MockKernel(KernelType.purity, integral_bounds=(0, 1))
    cl_abundance.add_kernel(p_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 0
        assert len(bounds) == 5
        assert bounds == [(13, 17), (0, 2), z_limits, mass_limits, (0, 1)]
        assert len(spi.arg_reader.extra_args) == 0
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
            KernelType.z_proxy.name: 2,
            KernelType.mass_proxy.name: 3,
            KernelType.purity.name: 4,
        }


def test_scipy_get_integration_bounds_analytic_slns(
    cl_abundance: ClusterAbundance,
):
    spi = ScipyIntegrator()

    z_array = np.linspace(0, 2, 10)
    m_array = np.linspace(13, 17, 10)
    z_bins = list(zip(z_array[:-1], z_array[1:]))
    m_bins = list(zip(m_array[:-1], m_array[1:]))

    a_kernel = MockKernel(KernelType.mass_proxy, has_analytic_sln=True)
    cl_abundance.add_kernel(a_kernel)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 1
        assert extra_args == [mass_limits]
        assert spi.arg_reader.extra_args == {KernelType.mass_proxy.name: 2}

        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    a_kernel2 = MockKernel(KernelType.z_proxy, has_analytic_sln=True)
    cl_abundance.add_kernel(a_kernel2)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 2
        assert extra_args == [mass_limits, z_limits]
        assert spi.arg_reader.extra_args == {
            KernelType.mass_proxy.name: 2,
            KernelType.z_proxy.name: 3,
        }

        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }

    a_kernel3 = MockKernel(
        KernelType.purity, has_analytic_sln=True, integral_bounds=(0, 1)
    )
    cl_abundance.add_kernel(a_kernel3)

    for z_limits, mass_limits in zip(z_bins, m_bins):
        bounds, extra_args = spi.get_integration_bounds(
            cl_abundance, z_limits, mass_limits
        )

        assert len(extra_args) == 3
        assert extra_args == [mass_limits, z_limits, (0, 1)]
        assert spi.arg_reader.extra_args == {
            KernelType.mass_proxy.name: 2,
            KernelType.z_proxy.name: 3,
            KernelType.purity.name: 4,
        }

        assert len(bounds) == 2
        assert bounds == [(13, 17), (0, 2)]
        assert spi.arg_reader.integral_bounds == {
            KernelType.mass.name: 0,
            KernelType.z.name: 1,
        }


def test_scipy_integrator_integrate():
    sci = ScipyIntegrator()

    def integrand(*int_args):
        x = int_args[0]
        return x

    bounds = [(0, 1)]
    result = sci.integrate(integrand, bounds, [])
    assert result == 0.5


def test_numcosmo_integrator_integrate():
    nci = NumCosmoIntegrator()

    def integrand(*int_args):
        x = int_args[0]
        return x

    bounds = [(0, 1)]
    result = nci.integrate(integrand, bounds, [])
    assert result == 0.5
