"""TwoPoint theory support."""

import copy
import numpy as np
from numpy import typing as npt

from firecrown.generators.two_point import EllOrThetaConfig, ELL_FOR_XI_DEFAULTS
from firecrown.likelihood.source import Source
from firecrown.metadata_types import TracerNames
from firecrown.updatable import Updatable
from firecrown.parameters import ParamsMap

SACC_DATA_TYPE_TO_CCL_KIND = {
    "galaxy_density_cl": "cl",
    "galaxy_density_xi": "NN",
    "galaxy_shearDensity_cl_e": "cl",
    "galaxy_shearDensity_xi_t": "NG",
    "galaxy_shear_cl_ee": "cl",
    "galaxy_shear_xi_minus": "GG-",
    "galaxy_shear_xi_plus": "GG+",
    "cmbGalaxy_convergenceDensity_xi": "NN",
    "cmbGalaxy_convergenceShear_xi_t": "NG",
}


class TwoPointTheory(Updatable):
    """Making predictions for TwoPoint statistics."""

    def __init__(
        self,
        *,
        sacc_data_type: str,
        sources: tuple[Source, Source],
        ell_or_theta_min: float | int | None = None,
        ell_or_theta_max: float | int | None = None,
    ) -> None:
        """Initialize a new TwoPointTheory object.

        :param sacc_data_type: the name of the SACC data type for this theory.
        :param source0: the first source
        :param source1: the second source
        """
        super().__init__()
        self.sacc_data_type = sacc_data_type
        self.ccl_kind: str = ""
        self.sources = sources
        self.ell_for_xi_config: dict[str, int] = {}
        self.ell_or_theta_config: None | EllOrThetaConfig = None
        self.ell_or_theta_min = ell_or_theta_min
        self.ell_or_theta_max = ell_or_theta_max
        self.window: None | npt.NDArray[np.float64] = None
        self.sacc_tracers: None | TracerNames = None
        self.ells: None | npt.NDArray[np.int64] = None
        self.thetas: None | npt.NDArray[np.float64] = None
        self.mean_ells: None | npt.NDArray[np.float64] = None
        self.ells_for_xi: None | npt.NDArray[np.int64] = None
        self.ell_for_xi_config = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        self.ell_or_theta_config = None
        self.window = None
        self.cells: dict[TracerNames, npt.NDArray[np.float64]] = {}

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

    def set_ccl_kind(self, sacc_data_type):
        """Set the CCL kind for this statistic."""
        self.sacc_data_type = sacc_data_type
        if self.sacc_data_type in SACC_DATA_TYPE_TO_CCL_KIND:
            self.ccl_kind = SACC_DATA_TYPE_TO_CCL_KIND[self.sacc_data_type]
        else:
            raise ValueError(f"The SACC data type {sacc_data_type} is not supported!")
