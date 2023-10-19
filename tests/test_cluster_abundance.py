import pytest
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.kernel import Kernel, KernelType, ArgReader
import pyccl
import numpy as np
from typing import List, Tuple
from firecrown.parameters import ParamsMap


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

    def distribution(self, args: List[float], args_map: ArgReader):
        return 1.0


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
    assert cl_abundance.cosmo is None
    cosmo = pyccl.CosmologyVanillaLCDM()
    pmap = ParamsMap()

    cl_abundance.update_ingredients(cosmo, pmap)
    assert cl_abundance.cosmo is not None
    assert cl_abundance.cosmo == cosmo


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
