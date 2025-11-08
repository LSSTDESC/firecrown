"""DES Y1 3x2pt likelihood factory with perturbation theory.

This module demonstrates how to build a DES Y1 3x2pt likelihood using
perturbation theory (PT) for modeling non-linear galaxy bias and intrinsic
alignments. This is a simplified example showing one source and one lens
bin with PT-based systematics.

Key differences from the standard template:
- Uses TattAlignmentSystematic for redshift-dependent intrinsic alignment
- Includes PTNonLinearBiasSystematic for perturbative galaxy bias modeling
- Configures EulerianPTCalculator for PT-based power spectrum calculations
- Enables redshift-space distortions (RSD) in number counts
"""

import os
import pyccl

from firecrown.likelihood.factories import load_sacc_data
from firecrown.likelihood.likelihood import NamedParameters
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory


def build_likelihood(params: NamedParameters) -> tuple[ConstGaussian, ModelingTools]:
    """Build DES Y1 3x2pt likelihood with perturbation theory modeling.

    This simplified example demonstrates PT-based modeling with:
    - One weak lensing source (src0) with Tatt IA systematic
    - One lens population (lens0) with PT non-linear bias
    - All four two-point correlation types (xi+, xi-, gamma_t, w_theta)

    Required parameters:
    - sacc_file: Path to SACC data file

    :param params: Named parameters containing configuration
    :return: Configured ConstGaussian likelihood and ModelingTools with PT
    :raises ValueError: If required parameters are missing
    :raises FileNotFoundError: If SACC file does not exist
    """
    # Tatt intrinsic alignment systematic with redshift dependence
    # Models IA as a power law in (1+z): A_IA * [(1+z)/(1+z_piv)]^eta
    # This is more flexible than LinearAlignmentSystematic
    ia_systematic = wl.TattAlignmentSystematic(include_z_dependence=True)

    # Photo-z shift for weak lensing source
    src_pzshift = wl.PhotoZShift(sacc_tracer="src0")

    # Weak lensing source with IA and photo-z systematics
    src0 = wl.WeakLensing(sacc_tracer="src0", systematics=[src_pzshift, ia_systematic])

    # Photo-z shift for lens population
    lens_pzshift = nc.PhotoZShift(sacc_tracer="lens0")

    # Magnification bias systematic (constant across redshift)
    magnification = nc.ConstantMagnificationBiasSystematic(sacc_tracer="lens0")

    # PT-based non-linear bias systematic
    # Models galaxy bias using perturbation theory with higher-order terms
    # Requires EulerianPTCalculator in ModelingTools
    nl_bias = nc.PTNonLinearBiasSystematic(sacc_tracer="lens0")

    # Number counts source with RSD enabled
    # has_rsd=True includes redshift-space distortions in clustering
    lens0 = nc.NumberCounts(
        sacc_tracer="lens0",
        has_rsd=True,
        systematics=[lens_pzshift, magnification, nl_bias],
    )

    # Cosmic shear two-point functions (xi+ and xi-)
    xip_src0_src0 = TwoPoint(
        source0=src0, source1=src0, sacc_data_type="galaxy_shear_xi_plus"
    )
    xim_src0_src0 = TwoPoint(
        source0=src0, source1=src0, sacc_data_type="galaxy_shear_xi_minus"
    )

    # Galaxy-galaxy lensing (tangential shear)
    gammat_lens0_src0 = TwoPoint(
        source0=lens0,
        source1=src0,
        sacc_data_type="galaxy_shearDensity_xi_t",
    )

    # Galaxy clustering (angular correlation function)
    wtheta_lens0_lens0 = TwoPoint(
        source0=lens0,
        source1=lens0,
        sacc_data_type="galaxy_density_xi",
    )

    # Configure Eulerian perturbation theory calculator
    # - with_NC=True: Enable number counts (galaxy clustering) PT terms
    # - with_IA=True: Enable intrinsic alignment PT terms
    # - log10k_min/max: k-space range for PT calculations [h/Mpc]
    # - nk_per_decade: Resolution of k-space sampling
    pt_calculator = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
    )

    # Create modeling tools with PT calculator
    # PT calculator is used for computing non-linear corrections
    # to power spectra beyond the standard Halofit approach
    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator, ccl_factory=CCLFactory(require_nonlinear_pk=True)
    )

    # Build Gaussian likelihood from two-point statistics
    # Statistics order determines data vector structure
    likelihood = ConstGaussian(
        statistics=[xip_src0_src0, xim_src0_src0, gammat_lens0_src0, wtheta_lens0_lens0]
    )

    # Load SACC data file (with environment variable expansion)
    sacc_file = params.get_string("sacc_file")
    sacc_file = os.path.expandvars(sacc_file)
    sacc_data = load_sacc_data(sacc_file)

    # Initialize likelihood with SACC data
    # - Two-point functions extract relevant correlation functions
    # - Sources receive corresponding redshift distributions
    # - Covariance matrix is loaded for parameter estimation
    likelihood.read(sacc_data)

    # Return likelihood and modeling tools for parameter estimation
    # Compatible with CosmoSIS, Cobaya and NumCosmo
    return likelihood, modeling_tools
