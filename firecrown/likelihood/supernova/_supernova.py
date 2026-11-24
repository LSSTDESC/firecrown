"""Type Ia supernova statistic support."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyccl
import sacc
from sacc.tracers import MiscTracer

from firecrown.data_types import DataVector, TheoryVector
from firecrown.likelihood_base import Statistic
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import register_new_updatable_parameter

SNIA_DEFAULT_M = -19.2


class Supernova(Statistic):
    """A statistic for Type Ia supernova apparent-magnitude measurements.

    The observed values are compared with apparent magnitudes predicted as
    :math:`m = M + \\mu`, where :math:`M` is the absolute magnitude and
    :math:`\\mu` is the cosmological distance modulus.

    Redshifts are read from the SACC data and converted to scale factors using
    :math:`a = 1 / (1 + z)`. Only the scale factors are retained internally.

    The statistic must be initialized by calling :meth:`read` before it is used.
    """

    def __init__(self, sacc_tracer: str) -> None:
        """Initialize a Type Ia supernova statistic.

        The named tracer must be a :class:`~sacc.tracers.MiscTracer`. The tracer
        name is also used as the parameter prefix, so the sampler-defined
        absolute-magnitude parameter is named ``<sacc_tracer>_M``. Its default
        value is :data:`SNIA_DEFAULT_M`.

        :param sacc_tracer: name of the supernova tracer in the SACC file.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.sacc_tracer = sacc_tracer
        self.data_vector: None | DataVector = None
        self.a: None | npt.NDArray[np.float64] = None
        self.M = register_new_updatable_parameter(default_value=SNIA_DEFAULT_M)

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the supernova measurements and redshifts from a SACC object.

        The SACC object must contain the configured
        :class:`~sacc.tracers.MiscTracer` and at least one data point with data
        type ``supernova_distance_mu``, the tracer tuple ``(sacc_tracer,)``, and a
        ``z`` tag. The data point values and scale factors are retained in the
        order in which SACC returns the matching data points.

        :param sacc_data: SACC data object containing the supernova measurements.
        :raises ValueError: if the named tracer is absent or is not a
            :class:`~sacc.tracers.MiscTracer`.
        :raises AssertionError: if no matching data points are present.
        """
        # Validate the tracer even though only its name is needed for data lookup.
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
        self.a = np.array(1.0 / (1.0 + z), dtype=np.float64)
        self.data_vector = DataVector.from_list([dp.value for dp in data_points])
        self.sacc_indices = np.arange(len(self.data_vector))
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        """Return the observed supernova apparent magnitudes.

        :returns: the data vector populated by :meth:`read`.
        :raises AssertionError: if :meth:`read` has not populated the data vector.
        """
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute predicted apparent magnitudes using CCL.

        This method uses the scale factors loaded by :meth:`read` and the current
        sampler value of ``M`` to evaluate :math:`m = M + \\mu` at each supernova.
        It is normally called through
        :meth:`~firecrown.likelihood_base.Statistic.compute_theory_vector` after
        the statistic has been read and updated with parameters.

        :param tools: modeling tools providing the CCL cosmology.
        :returns: predicted apparent magnitudes in SACC data-point order.
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        prediction = self.M + pyccl.distance_modulus(ccl_cosmo, self.a)
        return TheoryVector.create(prediction)
