"""Tests for the cluster abundance module."""
from unittest.mock import Mock
import pytest
import pyccl
import numpy as np
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.kernel import Kernel, KernelType
from firecrown.models.cluster.properties import ClusterProperty


@pytest.fixture(name="cluster_abundance")
def fixture_cluster_abundance():
    """Test fixture that represents an assembled cluster abundance class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterAbundance(13, 17, 0, 2, hmf)
    return ca


@pytest.fixture(name="dirac_delta_kernel")
def fixture_dirac_delta_kernel():
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=True,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    return mk


@pytest.fixture(name="integrable_kernel")
def fixture_integrable_kernel():
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS_PROXY,
        is_dirac_delta=False,
        has_analytic_sln=False,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    return mk


@pytest.fixture(name="analytic_sln_kernel")
def fixture_analytic_sln_solved_kernel():
    mk = Mock(
        spec=Kernel,
        kernel_type=KernelType.MASS,
        is_dirac_delta=False,
        has_analytic_sln=True,
    )
    mk.distribution.return_value = np.atleast_1d(1.0)
    return mk


@pytest.fixture(name="integrand_args")
def fixture_integrand_args():
    sky_area = 439.78986
    mass = np.linspace(13, 17, 5)
    z = np.linspace(0, 1, 5)
    mass_proxy = np.linspace(0, 5, 5)
    z_proxy = np.linspace(0, 1, 5)
    mass_proxy_limits = (0, 5)
    z_proxy_limits = (0, 1)
    return (mass, z, sky_area, mass_proxy, z_proxy, mass_proxy_limits, z_proxy_limits)


def test_cluster_abundance_init(cluster_abundance: ClusterAbundance):
    assert cluster_abundance is not None
    assert cluster_abundance.cosmo is None
    # pylint: disable=protected-access
    assert cluster_abundance._hmf_cache == {}
    assert isinstance(
        cluster_abundance.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )
    assert cluster_abundance.min_mass == 13.0
    assert cluster_abundance.max_mass == 17.0
    assert cluster_abundance.min_z == 0.0
    assert cluster_abundance.max_z == 2.0
    assert len(cluster_abundance.kernels) == 0


def test_cluster_update_ingredients(cluster_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)
    assert cluster_abundance.cosmo is not None
    assert cluster_abundance.cosmo == cosmo
    # pylint: disable=protected-access
    assert cluster_abundance._hmf_cache == {}


def test_cluster_add_integrable_kernel(
    cluster_abundance: ClusterAbundance, integrable_kernel: Kernel
):
    cluster_abundance.add_kernel(integrable_kernel)
    assert len(cluster_abundance.kernels) == 1
    assert isinstance(cluster_abundance.kernels[0], Kernel)

    assert len(cluster_abundance.analytic_kernels) == 0
    assert len(cluster_abundance.dirac_delta_kernels) == 0
    assert len(cluster_abundance.integrable_kernels) == 1


def test_cluster_add_dirac_delta_kernel(
    cluster_abundance: ClusterAbundance, dirac_delta_kernel: Kernel
):
    cluster_abundance.add_kernel(dirac_delta_kernel)
    assert len(cluster_abundance.kernels) == 1
    assert isinstance(cluster_abundance.kernels[0], Kernel)

    assert len(cluster_abundance.analytic_kernels) == 0
    assert len(cluster_abundance.dirac_delta_kernels) == 1
    assert len(cluster_abundance.integrable_kernels) == 0


def test_cluster_add_analytic_sln_kernel(
    cluster_abundance: ClusterAbundance, analytic_sln_kernel: Kernel
):
    cluster_abundance.add_kernel(analytic_sln_kernel)
    assert len(cluster_abundance.kernels) == 1
    assert isinstance(cluster_abundance.kernels[0], Kernel)

    assert len(cluster_abundance.analytic_kernels) == 1
    assert len(cluster_abundance.dirac_delta_kernels) == 0
    assert len(cluster_abundance.integrable_kernels) == 0


def test_abundance_comoving_returns_value(cluster_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)

    result = cluster_abundance.comoving_volume(np.linspace(0.1, 1, 10), 360**2)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 10
    assert np.all(result > 0)


@pytest.mark.regression
def test_abundance_massfunc_returns_value(cluster_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)

    result = cluster_abundance.mass_function(
        np.linspace(13, 17, 5), np.linspace(0.1, 1, 5)
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)


@pytest.mark.regression
def test_abundance_get_integrand(
    cluster_abundance: ClusterAbundance, integrable_kernel: Kernel, integrand_args
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)
    cluster_abundance.add_kernel(integrable_kernel)

    integrand = cluster_abundance.get_integrand()
    assert callable(integrand)
    result = integrand(*integrand_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_mass(
    cluster_abundance: ClusterAbundance, integrable_kernel: Kernel, integrand_args
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)
    cluster_abundance.add_kernel(integrable_kernel)

    integrand = cluster_abundance.get_integrand(average_properties=ClusterProperty.MASS)
    assert callable(integrand)
    result = integrand(*integrand_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_redshift(
    cluster_abundance: ClusterAbundance, integrable_kernel: Kernel, integrand_args
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)
    cluster_abundance.add_kernel(integrable_kernel)

    integrand = cluster_abundance.get_integrand(
        average_properties=ClusterProperty.REDSHIFT
    )
    assert callable(integrand)
    result = integrand(*integrand_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_mass_and_redshift(
    cluster_abundance: ClusterAbundance, integrable_kernel: Kernel, integrand_args
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)
    cluster_abundance.add_kernel(integrable_kernel)

    average_properties = ClusterProperty.MASS | ClusterProperty.REDSHIFT
    integrand = cluster_abundance.get_integrand(average_properties=average_properties)
    assert callable(integrand)
    result = integrand(*integrand_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)


@pytest.mark.regression
def test_abundance_get_integrand_avg_not_implemented_throws(
    cluster_abundance: ClusterAbundance, integrable_kernel: Kernel, integrand_args
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_abundance.update_ingredients(cosmo)
    cluster_abundance.add_kernel(integrable_kernel)

    average_properties = ClusterProperty.SHEAR
    integrand = cluster_abundance.get_integrand(average_properties=average_properties)
    assert callable(integrand)
    with pytest.raises(NotImplementedError):
        _ = integrand(*integrand_args)
