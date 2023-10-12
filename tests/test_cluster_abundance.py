from firecrown.models.cluster_abundance import ClusterAbundance
from firecrown.models.kernel import Kernel, KernelType
import pyccl
import numpy as np
from typing import List, Tuple, Dict


class MockKernel(Kernel):
    def __init__(self, integral_bounds: List[Tuple[float, float]] = None):
        super().__init__(KernelType.mass, integral_bounds)

    def distribution(self, args: List[float], index_lkp: Dict[str, int]):
        return 1.0


def test_cluster_abundance_init():
    hmf = pyccl.halos.MassFuncTinker08()
    ca = ClusterAbundance(hmf)
    assert ca is not None
    assert ca.cosmo is None
    assert isinstance(ca.halo_mass_function, pyccl.halos.MassFuncTinker08)


def test_cluster_update_ingredients():
    hmf = pyccl.halos.MassFuncTinker08()
    ca = ClusterAbundance(hmf)

    assert ca.cosmo is None
    cosmo = pyccl.CosmologyVanillaLCDM()

    ca.update_ingredients(cosmo)
    assert ca.cosmo is not None
    assert ca.cosmo == cosmo


def test_cluster_sky_area():
    hmf = pyccl.halos.MassFuncTinker08()
    ca = ClusterAbundance(hmf)
    assert ca.sky_area == 360.0**2
    assert ca.sky_area_rad == 4 * np.pi**2

    ca.sky_area = 180.0**2
    assert ca.sky_area == 180.0**2
    assert ca.sky_area_rad == np.pi**2


def test_cluster_add_kernel():
    hmf = pyccl.halos.MassFuncTinker08()
    ca = ClusterAbundance(hmf)
    assert len(ca.kernels) == 0

    ca.add_kernel(MockKernel())
    assert len(ca.kernels) == 1
    assert isinstance(ca.kernels[0], Kernel)


def test_cluster_abundance_bounds():
    hmf = pyccl.halos.MassFuncTinker08()
    ca = ClusterAbundance(hmf)
    bounds, index_lookup = ca.get_integration_bounds()
    assert len(bounds) == 0
    assert len(index_lookup) == 0

    ca.add_kernel(MockKernel())
    bounds, index_lookup = ca.get_integration_bounds()
    assert len(bounds) == 0
    assert len(index_lookup) == 0

    ca.add_kernel(MockKernel([(0, 1)]))
    bounds, index_lookup = ca.get_integration_bounds()
    assert len(bounds) == 1
    assert len(index_lookup) == 1
    assert index_lookup["mass"] == 0
    assert bounds[index_lookup["mass"]] == [(0, 1)]


# def test_cluster_abundance_integrand():
#     hmf = pyccl.halos.MassFuncTinker08()
#     ca = ClusterAbundance(hmf)
