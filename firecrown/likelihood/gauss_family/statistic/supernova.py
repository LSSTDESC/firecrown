from __future__ import annotations
from typing import Tuple, final
import functools

import numpy as np
import pyccl

from .statistic import Statistic
from ....parameters import ParamsMap, RequiredParameters

# only supported types are here, any thing else will throw
# a value error
SACC_DATA_TYPE_TO_CCL_KIND = {"supernova": "sn"}

Z_FOR_MU_DEFAULTS = dict(min=0, max=2, n=100)


def _z_for_mu(*, min, max, n):
    """Build an array of z to sample the distance modulus
    predictions.
    """
    return np.linspace(min, max, n)


@functools.lru_cache(maxsize=128)
def _cached_distmod(cosmo, tracers, z):
    a = 1.0 / (1 + z)
    return pyccl.background.distance_modulus(cosmo, *tracers, np.array(a))


class Supernova(Statistic):
    def __init__(self, sacc_tracer):
        self.sacc_tracer = sacc_tracer
        self.data_vector = None

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        data_points = sacc_data.get_data_points(
            data_type="supernova_distance_mu", tracers=(self.sacc_tracer,)
        )

        self.z = np.array([dp.get_tag("z") for dp in data_points])
        self.a = 1.0 / (1.0 + self.z)
        self.data_vector = np.array([dp.value for dp in data_points])
        self.sacc_inds = list(range(0, len(self.data_vector)))

    @final
    def _update(self, params: ParamsMap):
        self.M = params["m"]  # CosmoSIS makes everything lowercase

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(["m"])

    def compute(
        self, cosmo: pyccl.Cosmology
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a two-point statistic from sources.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        sources : dict
            A dictionary mapping sources to their objects. The sources must
            already have been rendered by calling `render` on them.
        systematics : dict, optional
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        theory_vector = self.M + pyccl.distance_modulus(cosmo, self.a)

        assert self.data_vector is not None

        return np.array(self.data_vector), np.array(theory_vector)
