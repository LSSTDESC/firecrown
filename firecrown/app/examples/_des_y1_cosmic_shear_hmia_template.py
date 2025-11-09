"""DES Y1 cosmic shear likelihood with halo model intrinsic alignment.

This module demonstrates cosmic shear analysis using a halo model approach
for intrinsic alignments with configurable halo mass function and bias.
"""

import os
import pyccl

from firecrown.likelihood.factories import load_sacc_data
from firecrown.likelihood.likelihood import NamedParameters
import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory


def build_likelihood(params: NamedParameters) -> tuple[ConstGaussian, ModelingTools]:
    """Build DES Y1 cosmic shear likelihood with halo model IA.

    Uses halo model approach for intrinsic alignments with configurable
    mass function, halo bias, and concentration-mass relation.

    Required parameters:
    - sacc_file: Path to SACC data file

    :param params: Named parameters containing configuration
    :return: Configured ConstGaussian likelihood and ModelingTools
    """
    sacc_file = os.path.expandvars(params.get_string("sacc_file"))
    sacc_data = load_sacc_data(sacc_file)

    # Create IA systematic using halo model
    ia_systematic = wl.HMAlignmentSystematic()

    # Create weak lensing source with photo-z shift and halo model IA
    pzshift = wl.PhotoZShift(sacc_tracer="src0")
    src0 = wl.WeakLensing(sacc_tracer="src0", systematics=[pzshift, ia_systematic])

    # Create two-point statistics for cosmic shear
    stats = [
        TwoPoint("galaxy_shear_xi_plus", src0, src0),
        TwoPoint("galaxy_shear_xi_minus", src0, src0),
    ]

    # Configure halo model components
    mass_def = "200m"
    hmc = pyccl.halos.HMCalculator(
        mass_function="Tinker10", halo_bias="Tinker10", mass_def=mass_def
    )

    modeling_tools = ModelingTools(
        hm_calculator=hmc,
        cM_relation="Duffy08",
        ccl_factory=CCLFactory(require_nonlinear_pk=True),
    )

    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)

    return likelihood, modeling_tools
