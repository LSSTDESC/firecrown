"""Supernova statistic support."""

from __future__ import annotations
from typing import Optional

import numpy as np
import numpy.typing as npt

import pyccl
import sacc
from sacc.tracers import MiscTracer

from ....modeling_tools import ModelingTools
from .statistic import Statistic, DataVector, TheoryVector
from .... import parameters


class PeculiarVel(Statistic):
    """A supernova statistic.

    This statistic that applies an additive shift M to a supernova's distance
    modulus.
    """

    def __init__(self, sacc_tracer: str) -> None:
        """Initialize this statistic."""
        super().__init__(parameter_prefix=sacc_tracer)

        self.sacc_tracer = sacc_tracer
        self.data_vector: Optional[DataVector] = None
        self.dv_len = 0

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file."""
        # We do not actually need the tracer, but we want to make sure the SACC
        # data contains this tracer.
        # TODO: remove the work-around when the new version of SACC supporting
        # sacc.Sacc.has_tracer is available.
        try:
            tracer = sacc_data.get_tracer(self.sacc_tracer)
        except KeyError as exc:
            # Translate to the error type we want
            raise ValueError(
                f"The SACC file does not contain the MiscTracer {self.sacc_tracer}"
            ) from exc
        if not isinstance(tracer, MiscTracer):
            raise ValueError(
                f"The SACC tracer {self.sacc_tracer} is not a " f"MiscTracer"
            )

        data_points = sacc_data.get_data_points(
            data_type="peculiar_vel", tracers=(self.sacc_tracer,)
        )
        self.data_vector = DataVector.from_list([dp.value for dp in data_points])
        self.dv_len = len(data_points)
        self.sacc_indices = np.arange(len(self.data_vector))
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Return dummy theory vector (not used in peculiar velocities."""
        assert self.dv_len is not 0
        return TheoryVector.from_list(np.zeros(self.dv_len))