"""Core TwoPointTheory class for making predictions."""

from collections.abc import Sequence

import numpy as np
import sacc
from numpy import typing as npt

from firecrown.generators.two_point import EllOrThetaConfig, LogLinearElls
from firecrown.likelihood._source import Source, Tracer
from firecrown.metadata_types import TracerNames
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.updatable import Updatable
from firecrown.utils import ClIntegrationOptions
from firecrown.models.two_point._interpolation import ApplyInterpolationWhen
from firecrown.models.two_point._sacc_utils import determine_ccl_kind


class TwoPointTheory(Updatable):
    """Making predictions for TwoPoint statistics."""

    def __init__(
        self,
        *,
        sacc_data_type: str,
        sources: tuple[Source, Source],
        interp_ells_gen: LogLinearElls = LogLinearElls(),
        ell_or_theta: None | EllOrThetaConfig = None,
        tracers: None | TracerNames = None,
        int_options: ClIntegrationOptions | None = None,
        apply_interp: ApplyInterpolationWhen = ApplyInterpolationWhen.DEFAULT,
    ) -> None:
        """Initialize a new TwoPointTheory object.

        :param sacc_data_type: the name of the SACC data type for this theory.
        :param sources: the sources for this theory; order matters
        :param interp_ells_gen: an object that will generate the values of
               the multipole order (values of ell) at which we will calculate
               "exact" C_ells, and which are then used to interpolate the values
               of C_ells.
        :param ell_or_theta: ell or theta configuration
        """
        super().__init__()
        self.sacc_data_type = sacc_data_type
        self.ccl_kind = determine_ccl_kind(sacc_data_type)
        self.sources = sources
        self.interp_ells_gen = interp_ells_gen
        self.ell_or_theta_config: None | EllOrThetaConfig = None
        self.window: None | npt.NDArray[np.float64] = None
        self.sacc_tracers = tracers
        self.ells: None | npt.NDArray[np.int64] = None
        self.thetas: None | npt.NDArray[np.float64] = None
        self.mean_ells: None | npt.NDArray[np.float64] = None
        self.cells: dict[TracerNames, npt.NDArray[np.float64]] = {}

        self.ells_for_xi: npt.NDArray[np.int64] = interp_ells_gen.generate()

        self.ell_or_theta_config = ell_or_theta
        self.int_options = int_options
        self.apply_interp = apply_interp

    @property
    def source0(self) -> Source:
        """Return the first source."""
        return self.sources[0]

    @property
    def source1(self) -> Source:
        """Return the second source."""
        return self.sources[1]

    def _update(self, params: ParamsMap) -> None:
        """Implementation of Updatable interface method `_update`.

        This is needed because of the tuple data member, which is not  updated
        automatically.
        """
        for s in self.sources:
            s.update(params)

    def _reset(self):
        """Implementation of Updatable interface method `_reset`.

        This is needed because of the tuple data member, which is not reset
        automatically.
        """
        for s in self.sources:
            s.reset()

    def initialize_sources(self, sacc_data: sacc.Sacc) -> None:
        """Initialize this TwoPointTheory's sources  and tracer names.

        :param sacc_data: The data in the from which we read the data.
        :return: The tracer names.
        """
        self.sources[0].read(sacc_data)
        if self.sources[0] is not self.sources[1]:
            self.sources[1].read(sacc_data)
        for s in self.sources:
            assert s is not None
            assert s.sacc_tracer is not None
        tracers = (s.sacc_tracer for s in self.sources)
        self.sacc_tracers = TracerNames(*tracers)

    def get_tracers_and_scales(
        self, tools: ModelingTools
    ) -> tuple[Sequence[Tracer], float, Sequence[Tracer], float]:
        """Get tracers and scales for both sources.

        :param tools: The modeling tools to use.
        :result: The tracers and scales for both sources.
        """
        tracers0 = self.source0.get_tracers(tools)
        scale0 = self.source0.get_scale()

        if self.source0 is self.source1:
            tracers1, scale1 = tracers0, scale0
        else:
            tracers1 = self.source1.get_tracers(tools)
            scale1 = self.source1.get_scale()

        return tracers0, scale0, tracers1, scale1

    def generate_ells_for_interpolation(self):
        """Generate ells for interpolation."""
        assert self.ells is not None
        min_ell = int(self.ells[0])
        max_ell = int(self.ells[-1])
        return self.interp_ells_gen.generate(min_ell, max_ell)
