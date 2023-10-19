import pytest
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.kernel import Kernel, KernelType, ArgReader
import pyccl
import numpy as np
from typing import List, Tuple
from firecrown.parameters import ParamsMap, create


@pytest.fixture()
def cl_abundance():
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterAbundance(13, 17, 0, 2, hmf, 360.0**2)
    return ca


class MockKernel(Kernel):
    def __init__(
        self,
        kernel_type: KernelType,
        is_dirac_delta: bool = False,
        has_analytic_sln: bool = False,
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        super().__init__(kernel_type, is_dirac_delta, has_analytic_sln, integral_bounds)
        self.param = create()

    def distribution(self, args: List[float], args_map: ArgReader):
        return 1.0


class MockArgsReader(ArgReader):
    def __init__(self):
        super().__init__()
        self.integral_bounds_idx = 0
        self.extra_args_idx = 1

    def get_integral_bounds(self, int_args, kernel_type: KernelType):
        bounds_values = int_args[self.integral_bounds_idx]
        return bounds_values[:, self.integral_bounds[kernel_type.name]]

    def get_extra_args(self, int_args, kernel_type: KernelType):
        extra_values = int_args[self.extra_args_idx]
        return extra_values[self.extra_args[kernel_type.name]]


def test_cluster_abundance_init(cl_abundance: ClusterAbundance):
    assert cl_abundance is not None
    assert cl_abundance.cosmo is None
    assert isinstance(cl_abundance.halo_mass_function, pyccl.halos.MassFuncBocquet16)
    assert cl_abundance.min_mass == 13.0
    assert cl_abundance.max_mass == 17.0
    assert cl_abundance.min_z == 0.0
    assert cl_abundance.max_z == 2.0
    assert cl_abundance.sky_area == 360.0**2
    assert cl_abundance.sky_area_rad == 4 * np.pi**2
    assert len(cl_abundance.kernels) == 0


def test_cluster_update_ingredients(cl_abundance: ClusterAbundance):
    mk = MockKernel(KernelType.mass_proxy)
    cl_abundance.add_kernel(mk)

    assert cl_abundance.cosmo is None
    assert mk.param is None

    pmap = ParamsMap({"param": 42})
    cosmo = pyccl.CosmologyVanillaLCDM()

    cl_abundance.update_ingredients(cosmo, pmap)
    assert cl_abundance.cosmo is not None
    assert cl_abundance.cosmo == cosmo
    assert mk.param == 42


def test_cluster_sky_area(cl_abundance: ClusterAbundance):
    assert cl_abundance.sky_area == 360.0**2
    assert cl_abundance.sky_area_rad == 4 * np.pi**2

    cl_abundance.sky_area = 180.0**2
    assert cl_abundance.sky_area == 180.0**2
    assert cl_abundance.sky_area_rad == np.pi**2


def test_cluster_add_kernel(cl_abundance: ClusterAbundance):
    assert len(cl_abundance.kernels) == 0

    cl_abundance.add_kernel(MockKernel(KernelType.mass))
    assert len(cl_abundance.kernels) == 1
    assert isinstance(cl_abundance.kernels[0], Kernel)
    assert cl_abundance.kernels[0].kernel_type == KernelType.mass

    assert len(cl_abundance.analytic_kernels) == 0
    assert len(cl_abundance.dirac_delta_kernels) == 0
    assert len(cl_abundance.integrable_kernels) == 1

    cl_abundance.add_kernel(MockKernel(KernelType.mass, True))
    assert len(cl_abundance.analytic_kernels) == 0
    assert len(cl_abundance.dirac_delta_kernels) == 1
    assert len(cl_abundance.integrable_kernels) == 1

    cl_abundance.add_kernel(MockKernel(KernelType.mass, False, True))
    assert len(cl_abundance.analytic_kernels) == 1
    assert len(cl_abundance.dirac_delta_kernels) == 1
    assert len(cl_abundance.integrable_kernels) == 1


def test_abundance_comoving_vol_accepts_float(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo, ParamsMap())

    result = cl_abundance.comoving_volume(0.1)
    assert isinstance(result, float)
    assert result > 0


def test_abundance_comoving_vol_accepts_array(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo, ParamsMap())

    result = cl_abundance.comoving_volume(np.linspace(0.1, 1, 10))
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 10
    assert np.all(result > 0)


def test_abundance_massfunc_accepts_float(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo, ParamsMap())

    result = cl_abundance.mass_function(13, 0.1)
    assert isinstance(result, float)
    assert result > 0


def test_abundance_massfunc_accepts_array(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo, ParamsMap())

    result = cl_abundance.mass_function(np.linspace(13, 17, 5), np.linspace(0.1, 1, 5))
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)


def test_abundance_get_integrand(cl_abundance: ClusterAbundance):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cl_abundance.update_ingredients(cosmo, ParamsMap())
    cl_abundance.add_kernel(MockKernel(KernelType.mass))

    arg_reader = MockArgsReader()
    arg_reader.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}

    mass = np.linspace(13, 17, 5)
    z = np.linspace(0, 1, 5)
    integral_bounds = np.array(list(zip(mass, z)))

    integrand = cl_abundance.get_integrand()
    assert callable(integrand)
    result = integrand(integral_bounds, [], arg_reader)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)

    integrand = cl_abundance.get_integrand(avg_mass=True)
    assert callable(integrand)
    result = integrand(integral_bounds, [], arg_reader)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)

    integrand = cl_abundance.get_integrand(avg_redshift=True)
    assert callable(integrand)
    result = integrand(integral_bounds, [], arg_reader)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)

    integrand = cl_abundance.get_integrand(avg_redshift=True, avg_mass=True)
    assert callable(integrand)
    result = integrand(integral_bounds, [], arg_reader)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
