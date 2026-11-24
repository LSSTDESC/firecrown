"""Supernova and cosmic shear likelihood factory for Firecrown analysis.

This module defines a likelihood factory function that returns a list of
likelihoods. The supernova and cosmic shear data sets are independent; Firecrown
sums the log-likelihoods of the list, which is equivalent to a single Gaussian
likelihood whose covariance has zero blocks between the two data sets.
"""

import itertools as it
from pathlib import Path

import firecrown.likelihood.supernova as sn
import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood import ConstGaussian, Likelihood, NamedParameters, TwoPoint
from firecrown.likelihood.factories import load_sacc_data
from firecrown.modeling_tools import CCLFactory, ModelingTools

# The template is copied as a standalone factory file and cannot import the
# cosmic shear template.
# pylint: disable=duplicate-code


def build_likelihood(
    params: NamedParameters,
) -> tuple[list[Likelihood], ModelingTools]:
    """Build the supernova and the cosmic shear likelihoods.

    Required parameters:
    - sn_sacc_file: Path to the supernova SACC data file
    - shear_sacc_file: Path to the cosmic shear SACC data file
    - n_bins: Number of tomographic redshift bins

    :param params: Named parameters containing configuration
    :returns: The list of the two likelihoods and the modeling tools
    """
    sn_sacc_file = Path(params.get_string("sn_sacc_file"))
    shear_sacc_file = Path(params.get_string("shear_sacc_file"))
    n_bins = params.get_int("n_bins")

    # Supernova likelihood: distance moduli with their own covariance.
    sn_likelihood = ConstGaussian(
        statistics=[sn.Supernova(sacc_tracer="sn_ddf_sample")]
    )
    sn_likelihood.read(load_sacc_data(sn_sacc_file))

    # Cosmic shear likelihood: all auto and cross spectra of the tomographic bins,
    # each bin with a photo-z shift.
    sources = [
        wl.WeakLensing(
            sacc_tracer=f"trc{i}",
            systematics=[wl.PhotoZShift(sacc_tracer=f"trc{i}")],
        )
        for i in range(n_bins)
    ]
    shear_likelihood = ConstGaussian(
        statistics=[
            TwoPoint("galaxy_shear_cl_ee", source_x, source_y)
            for source_x, source_y in it.combinations_with_replacement(sources, 2)
        ]
    )
    shear_likelihood.read(load_sacc_data(shear_sacc_file))

    # One ModelingTools serves both likelihoods; cosmic shear requires the
    # nonlinear matter power spectrum.
    modeling_tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True))

    # Firecrown wraps the list in a SumLikelihood.
    return [sn_likelihood, shear_likelihood], modeling_tools
