"""Supernova statistic support
"""

from __future__ import annotations
from typing import Optional, Tuple, final
import functools

import numpy as np

import pyccl
import sacc

from .statistic import Statistic
from ....parameters import ParamsMap, RequiredParameters, DerivedParameterCollection


class Supernova(Statistic):
    """A statistic that applies an additive shift M to a supernova's distance
    modulus."""

    def __init__(self, sacc_tracer):
        """Initialize this statistic."""
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.data_vector = None
        self.a: Optional[np.ndarray] = None  # pylint: disable-msg=invalid-name
        self.M: Optional[np.ndarray] = None  # pylint: disable-msg=invalid-name

    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this statistic from the SACC file."""

        data_points = sacc_data.get_data_points(
            data_type="supernova_distance_mu", tracers=(self.sacc_tracer,)
        )
        # pylint: disable-next=invalid-name
        z = np.array([dp.get_tag("z") for dp in data_points])
        self.a = 1.0 / (1.0 + z)
        self.data_vector = np.array([dp.value for dp in data_points])
        self.sacc_inds = list(range(0, len(self.data_vector)))

    @final
    def _update(self, params: ParamsMap):
        self.M = params["m"]  # CosmoSIS makes everything lowercase

    @final
    def _reset(self):
        pass

    @final
    def required_parameters(self) -> RequiredParameters:
        """Return a RequiredParameters object containing the information for this
        statistic.

        The only required parameter is`m`.
        """
        return RequiredParameters(["m"])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def compute(self, cosmo: pyccl.Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a two-point statistic from sources."""

        theory_vector = self.M + pyccl.distance_modulus(cosmo, self.a)

        assert self.data_vector is not None

        return np.array(self.data_vector), np.array(theory_vector)
