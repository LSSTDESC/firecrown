"""Tests for the cluster abundance module."""
from unittest.mock import Mock
import pytest
import pyccl
import numpy as np
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.kernel import Kernel, KernelType
from firecrown.models.cluster.properties import ClusterProperty


@pytest.fixture(name="cl_abundance")
def fixture_cl_abundance():
    """Test fixture that represents an assembled cluster abundance class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterAbundance(13, 17, 0, 2, hmf)
    return ca


def test_cluster_abundance_init(cl_abundance: ClusterAbundance):
    assert cl_abundance is not None
    assert cl_abundance.cosmo is None
    assert isinstance(cl_abundance.halo_mass_function, pyccl.halos.MassFuncBocquet16)
    assert cl_abundance.min_mass == 13.0
    assert cl_abundance.max_mass == 17.0
    assert cl_abundance.min_z == 0.0
    assert cl_abundance.max_z == 2.0
    assert len(cl_abundance.kernels) == 0


def test_cluster_update_ingredients(cl_abundance: ClusterAbundance):
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        is_dirac_delta=True,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)

    assert cl_abundance.cosmo is None
    # pylint: disable=protected-access
    assert cl_abundance._hmf_cache == {}

    cosmo = pyccl.CosmologyVanillaLCDM()

    cl_abundance.update_ingredients(cosmo)
    assert cl_abundance.cosmo is not None
    assert cl_abundance.cosmo == cosmo
    # pylint: disable=protected-access
    assert cl_abundance._hmf_cache == {}

    cl_abundance.update_ingredients(None)
    assert cl_abundance.cosmo is None
    cl_abundance.update_ingredients(cosmo)
    assert cl_abundance.cosmo is not None
    assert cl_abundance.cosmo == cosmo


def test_cluster_add_kernel(cl_abundance: ClusterAbundance):
    assert len(cl_abundance.kernels) == 0

    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)
    assert len(cl_abundance.kernels) == 1
    assert isinstance(cl_abundance.kernels[0], Kernel)
    assert cl_abundance.kernels[0].kernel_type == KernelType.MASS

    assert len(cl_abundance.analytic_kernels) == 0
    assert len(cl_abundance.dirac_delta_kernels) == 0
    assert len(cl_abundance.integrable_kernels) == 1

    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=True,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)
    assert len(cl_abundance.analytic_kernels) == 0
    assert len(cl_abundance.dirac_delta_kernels) == 1
    assert len(cl_abundance.integrable_kernels) == 1

    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=True,
    )
    cl_abundance.add_kernel(mk)
    assert len(cl_abundance.analytic_kernels) == 1
    assert len(cl_abundance.dirac_delta_kernels) == 1
    assert len(cl_abundance.integrable_kernels) == 1


def test_abundance_comoving_vol_accepts_array(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo)

    result = cl_abundance.comoving_volume(np.linspace(0.1, 1, 10), 360**2)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 10
    assert np.all(result > 0)


@pytest.mark.regression
def test_abundance_massfunc_accepts_array(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo)

    result = cl_abundance.mass_function(np.linspace(13, 17, 5), np.linspace(0.1, 1, 5))
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)


@pytest.mark.regression
def test_abundance_get_integrand(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo)
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)

    sky_area = 439.78986
    mass = np.linspace(13, 17, 5)
    z = np.linspace(0, 1, 5)
    mass_proxy = np.linspace(0, 5, 5)
    z_proxy = np.linspace(0, 1, 5)
    mass_proxy_limits = (0, 5)
    z_proxy_limits = (0, 1)

    integrand = cl_abundance.get_integrand()
    assert callable(integrand)
    result = integrand(
        mass, z, sky_area, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_mass(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo)
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)

    sky_area = 439.78986
    mass = np.linspace(13, 17, 5)
    z = np.linspace(0, 1, 5)
    mass_proxy = np.linspace(0, 5, 5)
    z_proxy = np.linspace(0, 1, 5)
    mass_proxy_limits = (0, 5)
    z_proxy_limits = (0, 1)

    integrand = cl_abundance.get_integrand(average=ClusterProperty.MASS)
    assert callable(integrand)
    result = integrand(
        mass, z, sky_area, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_redshift(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo)
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)

    sky_area = 439.78986
    mass = np.linspace(13, 17, 5)
    z = np.linspace(0, 1, 5)
    mass_proxy = np.linspace(0, 5, 5)
    z_proxy = np.linspace(0, 1, 5)
    mass_proxy_limits = (0, 5)
    z_proxy_limits = (0, 1)

    integrand = cl_abundance.get_integrand(average=ClusterProperty.REDSHIFT)
    assert callable(integrand)
    result = integrand(
        mass, z, sky_area, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_mass_and_redshift(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo)
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)

    sky_area = 439.78986
    mass = np.linspace(13, 17, 5)
    z = np.linspace(0, 1, 5)
    mass_proxy = np.linspace(0, 5, 5)
    z_proxy = np.linspace(0, 1, 5)
    mass_proxy_limits = (0, 5)
    z_proxy_limits = (0, 1)

    average_properties = ClusterProperty.MASS | ClusterProperty.REDSHIFT
    integrand = cl_abundance.get_integrand(average=average_properties)
    assert callable(integrand)
    result = integrand(
        mass, z, sky_area, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_not_implemented_throws(
    cl_abundance: ClusterAbundance,
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo)
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    cl_abundance.add_kernel(mk)

    sky_area = 439.78986
    mass = np.linspace(13, 17, 5)
    z = np.linspace(0, 1, 5)
    mass_proxy = np.linspace(0, 5, 5)
    z_proxy = np.linspace(0, 1, 5)
    mass_proxy_limits = (0, 5)
    z_proxy_limits = (0, 1)

    average_properties = ClusterProperty.SHEAR
    integrand = cl_abundance.get_integrand(average=average_properties)
    assert callable(integrand)
    with pytest.raises(NotImplementedError):
        _ = integrand(
            mass, z, sky_area, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits
        )
