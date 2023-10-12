"""Likelihood factory function for cluster number counts."""

import os
import numpy as np

import pyccl as ccl
import sacc

from firecrown.likelihood.gauss_family.statistic.cluster_number_counts import (
    ClusterNumberCounts,
    DataVector,
)
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster_abundance import ClusterAbundance
from firecrown.models.sacc_adapter import SaccAdapter
from firecrown.models.mass_observable import MassRichnessMuSigma
from firecrown.models.redshift import SpectroscopicRedshift, DESY1PhotometricRedshift
from firecrown.models.kernel import Completeness, Purity
from firecrown.parameters import ParamsMap


def build_likelihood(build_parameters):
    """
    Here we instantiate the number density (or mass function) object.
    """

    # Pull params for the likelihood from build_parameters
    use_cluster_counts = build_parameters.get_bool("use_cluster_counts", True)
    use_mean_log_mass = build_parameters.get_bool("use_mean_log_mass", False)

    # Read in sacc data
    sacc_file_nm = "cluster_redshift_richness_sacc_data.fits"
    survey_name = "numcosmo_simulated_redshift_richness"
    sacc_path = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/cluster_number_counts/")
    )
    sacc_file = os.path.join(sacc_path, sacc_file_nm)

    sacc_data = sacc.Sacc.load_fits(sacc_file)
    counts = ClusterNumberCounts(use_cluster_counts, use_mean_log_mass, survey_name)

    hmf = ccl.halos.MassFuncTinker08()
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_abundance = ClusterAbundance(min_mass, max_mass, min_z, max_z, hmf)

    # Create and add the kernels you want in your cluster abundance
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    mass_observable_kernel = MassRichnessMuSigma(pivot_mass, pivot_redshift)
    # redshift_proxy_kernel = SpectroscopicRedshift()
    redshift_proxy_kernel = DESY1PhotometricRedshift()
    # completeness_kernel = Completeness()
    # purity_kernel = Purity()

    cluster_abundance.add_kernel(mass_observable_kernel)
    cluster_abundance.add_kernel(redshift_proxy_kernel)
    # cluster_abundance.add_kernel(completeness_kernel)
    # cluster_abundance.add_kernel(purity_kernel)

    # Put it all together
    stats_list = [counts]
    lk = ConstGaussian(stats_list)
    lk.read(sacc_data)
    modeling = ModelingTools(cluster_abundance=cluster_abundance)

    return lk, modeling
