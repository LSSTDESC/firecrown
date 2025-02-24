"""TwoPoint theory support."""

import copy
from typing import Sequence
import numpy as np
from numpy import typing as npt
import pyccl
import sacc

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


def calculate_pk(
    pk_name: str, tools: ModelingTools, tracer0: Tracer, tracer1: Tracer
) -> pyccl.Pk2D:
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
            # TODO: What if we are doing GGL, and need galaxies as tracers?
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
        # Fix a_arr because normalization is zero for a<~0.07
        # FIXME: Is this enough?
        a_arr = np.linspace(0.1, 1, 16)
        ccl_cosmo = tools.get_ccl_cosmology()
        hm_calculator = tools.get_hm_calculator()
        IA_bias_exponent = (
            2  # Square IA bias if both tracers are HM (doing II correlation).
        )
        if not (tracer0.has_hm and tracer1.has_hm):
            IA_bias_exponent = (
                1  # IA bias if not both tracers are HM (doing GI correlation).
            )
            if "galaxies" in [tracer0.field, tracer1.field]:
                other_profile = pyccl.halos.HaloProfileHOD(
                    mass_def=tools.hm_definition,
                    concentration=tools.get_cM_relation()
                )
            else:
                other_profile = pyccl.halos.HaloProfileNFW(
                    mass_def=tools.hm_definition,
                    concentration=tools.get_cM_relation(),
                    truncated=True,
                    fourier_analytic=True,
                )
            other_profile.ia_a_2h = (
                -1.0
            )  # used in GI contribution, which is negative.
            if not tracer0.has_hm:
                profile0 = other_profile
                profile1 = tracer1.halo_profile
            else:
                profile0 = tracer0.halo_profile
                profile1 = other_profile
        else:
            profile0 = tracer0.halo_profile
            profile1 = tracer1.halo_profile
        # Compute here the 1-halo power spectrum
        pk_1h = pyccl.halos.halomod_Pk2D(
            cosmo=ccl_cosmo,
            hmc=hm_calculator,
            prof=profile0,
            prof2=profile1,
            a_arr=a_arr,
            get_2h=False,
        )
        # Compute here the 2-halo power spectrum
        C1rhocrit = (
            5e-14 * pyccl.physical_constants.RHO_CRITICAL
        )  # standard IA normalisation
        pk_2h = pyccl.Pk2D.from_function(
            pkfunc=lambda k, a: profile0.ia_a_2h
            * profile1.ia_a_2h
            * (C1rhocrit * ccl_cosmo["Omega_m"] / ccl_cosmo.growth_factor(a))
            ** IA_bias_exponent
            * ccl_cosmo.nonlin_matter_power(k, a),
            is_logp=False,
        )
        pk = pk_1h + pk_2h
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
        tracers: None | TracerNames = None,
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
        self.sacc_tracers = tracers
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
