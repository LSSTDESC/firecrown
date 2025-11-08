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


def _build_sources() -> tuple[wl.WeakLensing, nc.NumberCounts]:
    """Create weak lensing and number counts sources with systematics.

    :return: Tuple of (weak lensing source, number counts source)
    """
    ia_systematic = wl.TattAlignmentSystematic(include_z_dependence=True)
    src_pzshift = wl.PhotoZShift(sacc_tracer="src0")
    src0 = wl.WeakLensing(sacc_tracer="src0", systematics=[src_pzshift, ia_systematic])

    lens_pzshift = nc.PhotoZShift(sacc_tracer="lens0")
    magnification = nc.ConstantMagnificationBiasSystematic(sacc_tracer="lens0")
    nl_bias = nc.PTNonLinearBiasSystematic(sacc_tracer="lens0")
    lens0 = nc.NumberCounts(
        sacc_tracer="lens0",
        has_rsd=True,
        systematics=[lens_pzshift, magnification, nl_bias],
    )

    return src0, lens0


def _build_two_point_statistics(
    src0: wl.WeakLensing, lens0: nc.NumberCounts
) -> list[TwoPoint]:
    """Create two-point correlation statistics.

    :param src0: Weak lensing source
    :param lens0: Number counts source
    :return: List of two-point statistics
    """
    return [
        TwoPoint(source0=src0, source1=src0, sacc_data_type="galaxy_shear_xi_plus"),
        TwoPoint(source0=src0, source1=src0, sacc_data_type="galaxy_shear_xi_minus"),
        TwoPoint(
            source0=lens0, source1=src0, sacc_data_type="galaxy_shearDensity_xi_t"
        ),
        TwoPoint(source0=lens0, source1=lens0, sacc_data_type="galaxy_density_xi"),
    ]


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
    # Create sources with systematics
    src0, lens0 = _build_sources()

    # Create two-point statistics
    statistics = _build_two_point_statistics(src0, lens0)

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
    likelihood = ConstGaussian(statistics=statistics)

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
