import numpy as np
from firecrown.parameters import ParamsMap
from firecrown.models.kernel import Kernel


class Redshift(Kernel):
    def __init__(self, params: ParamsMap = None):
        super().__init__()
        self.params = params

    def probability(self, mass, z, mass_proxy, z_proxy):
        pass


class SpectroscopicRedshiftUncertainty(Redshift):
    def __init__(self, params: ParamsMap = None):
        super().__init__(params)

    def probability(self, mass, z, mass_proxy, z_proxy):
        return 1.0


class DESY1PhotometricRedshiftUncertainty(Redshift):
    def __init__(self, params: ParamsMap = None):
        super().__init__(params)
        self.sigma_0 = 0.05

    def probability(self, mass, z, mass_proxy, z_proxy):
        sigma_z = self.sigma_0 * (1 + z)
        prefactor = 1 / (np.sqrt(2.0 * np.pi) * sigma_z)
        distribution = np.exp(-(1 / 2) * ((z_proxy - z) / sigma_z) ** 2.0)
        return prefactor * distribution
