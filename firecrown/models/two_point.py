"""TwoPoint theory support."""

import copy
import numpy as np
from numpy import typing as npt
import pyccl

from firecrown.generators.two_point import EllOrThetaConfig, ELL_FOR_XI_DEFAULTS
from firecrown.likelihood.source import Source, Tracer
from firecrown.metadata_types import TracerNames
from firecrown.updatable import Updatable
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools


def determine_ccl_kind(sacc_data_type: str) -> str:
    """Determine the CCL kind for this SACC data type.

    :param sacc_data_type: the name of the SACC data type
    :return: the CCL kind
    """
    match sacc_data_type:
        case "galaxy_density_cl" | "galaxy_shearDensity_cl_e" | "galaxy_shear_cl_ee":
            result = "cl"
        case "galaxy_density_xi":
            result = "NN"
        case "galaxy_shearDensity_xi_t":
            result = "NG"
        case "galaxy_shear_xi_minus":
            result = "GG-"
        case "galaxy_shear_xi_plus":
            result = "GG+"
        case "cmbGalaxy_convergenceDensity_xi":
            result = "NN"
        case "cmbGalaxy_convergenceShear_xi_t":
            result = "NG"
        case _:
            raise ValueError(f"The SACC data type {sacc_data_type} is not supported!")
    return result


def calculate_pk(pk_name: str, tools: ModelingTools, tracer0: Tracer, tracer1: Tracer):
    """Return the power spectrum named by pk_name.

    If the modeling tools already has the power spectrum, it is returned.
    If not, is is computed with the help of the modeling tools.

    :param pk_name: The name of the power spectrum to return.
    :param tools: The modeling tools to use.
    :param tracer0: The first tracer to use.
    :param tracer1: The second tracer to use.
    :return: The power spectrum.
    """
    if tools.has_pk(pk_name):
        # Use existing power spectrum
        pk = tools.get_pk(pk_name)
    elif tracer0.has_pt or tracer1.has_pt:
        if not (tracer0.has_pt and tracer1.has_pt):
            # Mixture of PT and non-PT tracers
            # Create a dummy matter PT tracer for the non-PT part
            matter_pt_tracer = pyccl.nl_pt.PTMatterTracer()
            if not tracer0.has_pt:
                tracer0.pt_tracer = matter_pt_tracer
            else:
                tracer1.pt_tracer = matter_pt_tracer
        # Compute perturbation power spectrum

        pt_calculator = tools.get_pt_calculator()
        pk = pt_calculator.get_biased_pk2d(
            tracer1=tracer0.pt_tracer,
            tracer2=tracer1.pt_tracer,
        )
    elif tracer0.has_hm or tracer1.has_hm:
        # Compute halo model power spectrum
        raise NotImplementedError("Halo model power spectra not supported yet")
    else:
        raise ValueError(f"No power spectrum for {pk_name} can be found.")
    return pk


class TwoPointTheory(Updatable):
    """Making predictions for TwoPoint statistics."""

    def __init__(
        self,
        *,
        sacc_data_type: str,
        sources: tuple[Source, Source],
        ell_or_theta_min: float | int | None = None,
        ell_or_theta_max: float | int | None = None,
        ell_for_xi: None | dict[str, int] = None,
        ell_or_theta: None | EllOrThetaConfig = None,
    ) -> None:
        """Initialize a new TwoPointTheory object.

        :param sacc_data_type: the name of the SACC data type for this theory.
        :param sources: the sources for this theory; order matters
        :param ell_or_theta_min: minimum ell for xi
        :param ell_or_theta_max: maximum ell for xi
        :param ell_for_xi: ell for xi configuration
        :param ell_or_theta: ell or theta configuration
        """
        super().__init__()
        self.sacc_data_type = sacc_data_type
        self.ccl_kind = determine_ccl_kind(sacc_data_type)
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
        self.cells: dict[TracerNames, npt.NDArray[np.float64]] = {}
        if ell_for_xi is not None:
            self.ell_for_xi_config.update(ell_for_xi)

        self.ell_or_theta_config = ell_or_theta

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
