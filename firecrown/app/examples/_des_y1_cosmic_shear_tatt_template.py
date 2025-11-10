"""DES Y1 cosmic shear likelihood with TATT intrinsic alignment model.

This module demonstrates cosmic shear analysis using the Tidal Alignment and
Tidal Torquing (TATT) model for intrinsic alignments with perturbation theory.
"""

import os
import pyccl.nl_pt

from firecrown.likelihood.factories import load_sacc_data
from firecrown.likelihood.likelihood import NamedParameters
import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory

# pylint: disable=duplicate-code


def build_likelihood(params: NamedParameters) -> tuple[ConstGaussian, ModelingTools]:
    """Build DES Y1 cosmic shear likelihood with TATT IA model.

    Uses the Tidal Alignment and Tidal Torquing (TATT) model for intrinsic
    alignments with redshift-dependent parameters and perturbation theory.

    Required parameters:
    - sacc_file: Path to SACC data file

    :param params: Named parameters containing configuration
    :return: Configured ConstGaussian likelihood and ModelingTools with PT
    """
    sacc_file = os.path.expandvars(params.get_string("sacc_file"))
    sacc_data = load_sacc_data(sacc_file)

    # Create IA systematic with redshift dependence
    ia_systematic = wl.TattAlignmentSystematic(include_z_dependence=True)

    # Create weak lensing source with photo-z shift and TATT IA
    pzshift = wl.PhotoZShift(sacc_tracer="src0")
    src0 = wl.WeakLensing(sacc_tracer="src0", systematics=[pzshift, ia_systematic])

    # Create two-point statistics for cosmic shear
    stats = [
        TwoPoint("galaxy_shear_xi_plus", src0, src0),
        TwoPoint("galaxy_shear_xi_minus", src0, src0),
    ]

    # Configure PT calculator for IA modeling
    pt_calculator = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=False,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
    )

    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator, ccl_factory=CCLFactory(require_nonlinear_pk=True)
    )

    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)

    return likelihood, modeling_tools
