import copy
import functools
import warnings

import numpy as np
import pyccl as ccl

from ..core import Statistic

# only supported types are here, any thing else will throw
# a value error
SACC_DATA_TYPE_TO_CCL_KIND = {
    "supernova": 'sn'
}

Z_FOR_MU_DEFAULTS = dict(min=0, max=2, n=100)

def _z_for_mu(*, min, max, n):
    """Build an array of z to sample the distance modulus
    predictions.
    """
    return  np.linspace(min, max,n)

@functools.lru_cache(maxsize=128)
def _cached_distmod(cosmo, tracers, z):
    a = 1./(1+z)
    return ccl.background.distance_modulus(cosmo, *tracers, np.array(a))

class SupernovaStatistic(Statistic):
    def __init__(self, sacc_tracer):
        self.sacc_tracers = sacc_tracer
        print(self.sacc_tracers)
        self.M = -19 #params['m'] # CosmoSIS makes everything lowercase
        print('SELF.M = ',self.M)
    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        tracer = sacc_data.get_tracer(self.sacc_tracers)
        data_points = sacc_data.get_data_points (data_type="supernova_distance_mu", tracers=(self.sacc_tracers,))

        self.z = np.array ([dp.get_tag ("z") for dp in data_points])
        self.a = 1.0 / (1.0 + self.z)
        self.m = np.array ([dp.value for dp in data_points])
        self.sacc_inds = list (range (0, len (self.m)))
        
    def update_params(self, params):
        self.M = params['m'] # CosmoSIS makes everything lowercase

    def compute(self, cosmo, params, sources, systematics=None):
        """Compute a two-point statistic from sources.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        sources : dict
            A dictionary mapping sources to their objects. The sources must
            already have been rendered by calling `render` on them.
        systematics : dict, optional
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        self.predicted_statistic_ = self.M + ccl.distance_modulus(cosmo, self.a)
        self.measured_statistic_ = self.m
        
