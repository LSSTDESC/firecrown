"""Supernova statistic support
"""

from __future__ import annotations
from typing import Optional, final

import numpy as np

import pyccl
import sacc

from ....likelihood.likelihood import Cosmology

from ....likelihood.likelihood import Cosmology
from .statistic import Statistic, DataVector, TheoryVector
from .... import parameters
from ....parameters import ParamsMap, RequiredParameters, DerivedParameterCollection


class Supernova(Statistic):
    """A statistic that applies an additive shift M to a supernova's distance
    modulus."""

    def __init__(self, sacc_tracer) -> None:
        """Initialize this statistic."""
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.data_vector: Optional[DataVector] = None
        self.a: Optional[np.ndarray] = None  # pylint: disable-msg=invalid-name
        self.M = parameters.create()  # pylint: disable-msg=invalid-name

    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this statistic from the SACC file."""

        data_points = sacc_data.get_data_points(
            data_type="supernova_distance_mu", tracers=(self.sacc_tracer,)
        )
        # pylint: disable-next=invalid-name
        z = np.array([dp.get_tag("z") for dp in data_points])
        self.a = 1.0 / (1.0 + z)
        self.data_vector = DataVector.from_list([dp.value for dp in data_points])
        self.sacc_indices = list(
            range(0, len(self.data_vector))
        )  # pylint: disable-msg=invalid-name

    @final
    def _update(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation has nothing to do."""

    @final
    def _reset(self):
        """Reset this statistic. This implementation has nothing to do."""
        pass

    @final
    def _required_parameters(self) -> RequiredParameters:
        """Return an empty RequiredParameters."""
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return an empty DerivedParameterCollection."""
        return DerivedParameterCollection([])

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, cosmo: Cosmology) -> TheoryVector:
        """Compute SNIa distance statistic using CCL."""
        prediction = self.M + pyccl.distance_modulus(cosmo.ccl_cosmo, self.a)
        return TheoryVector.create(prediction)
