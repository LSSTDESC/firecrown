"""Supernova statistic support."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyccl
import sacc
from sacc.tracers import MiscTracer

# firecrown is needed for backward compatibility; remove support for deprecated
# directory structure is removed.
import firecrown  # pylint: disable=unused-import # noqa: F401
from firecrown import parameters
from firecrown.likelihood.statistic import (
    DataVector,
    Statistic,
    TheoryVector,
)
from firecrown.modeling_tools import ModelingTools

SNIA_DEFAULT_M = -19.2


class Supernova(Statistic):
    """A supernova statistic.

    This statistic that applies an additive shift M to a supernova's distance
    modulus.
    """

    def __init__(self, sacc_tracer: str) -> None:
        """Initialize this statistic."""
        super().__init__(parameter_prefix=sacc_tracer)

        self.sacc_tracer = sacc_tracer
        self.data_vector: None | DataVector = None
        self.a: None | npt.NDArray[np.float64] = None
        self.M = parameters.register_new_updatable_parameter(
            default_value=SNIA_DEFAULT_M
        )

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
            data_type="supernova_distance_mu", tracers=(self.sacc_tracer,)
        )
        z = np.array([dp.get_tag("z") for dp in data_points])
        self.a = 1.0 / (1.0 + z)
        self.data_vector = DataVector.from_list([dp.value for dp in data_points])
        self.sacc_indices = np.arange(len(self.data_vector))
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute SNIa distance statistic using CCL."""
        ccl_cosmo = tools.get_ccl_cosmology()
        prediction = self.M + pyccl.distance_modulus(ccl_cosmo, self.a)
        return TheoryVector.create(prediction)
