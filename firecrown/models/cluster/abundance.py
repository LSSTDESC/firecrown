from pyccl.cosmology import Cosmology
from firecrown.models.cluster_theory import abundance as theo_abundance
from firecrown.parameters import ParamsMap


class ClusterAbundance(theo_abundance.ClusterAbundance):
    def update_ingredients(self, cosmo: Cosmology, params: ParamsMap):
        self._cosmo = cosmo
        self._hmf_cache = {}
        for kernel in self.kernels:
            kernel.pars.update(params)
