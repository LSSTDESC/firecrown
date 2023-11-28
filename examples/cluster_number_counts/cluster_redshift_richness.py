"""Likelihood factory function for cluster number counts."""

import os

import pyccl as ccl
import sacc

from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.statistic.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.kernel import (
    SpectroscopicRedshift,
)
from firecrown.models.cluster.mass_proxy import MurataBinned
from firecrown.likelihood.likelihood import NamedParameters, Likelihood
from typing import Tuple


def get_cluster_abundance() -> ClusterAbundance:
    hmf = ccl.halos.MassFuncBocquet16()
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_abundance = ClusterAbundance(min_mass, max_mass, min_z, max_z, hmf)

    # Create and add the kernels you want in your cluster abundance
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    mass_observable_kernel = MurataBinned(pivot_mass, pivot_redshift)
    cluster_abundance.add_kernel(mass_observable_kernel)

    redshift_proxy_kernel = SpectroscopicRedshift()
    cluster_abundance.add_kernel(redshift_proxy_kernel)

    return cluster_abundance


def build_likelihood(
    build_parameters: NamedParameters,
) -> Tuple[Likelihood, ModelingTools]:
    """
    Here we instantiate the number density (or mass function) object.
    """
    integrator = NumCosmoIntegrator()
    # integrator = ScipyIntegrator()

    # Pull params for the likelihood from build_parameters
    average_properties = ClusterProperty.NONE
    if build_parameters.get_bool("use_cluster_counts", True):
        average_properties |= ClusterProperty.COUNTS
    if build_parameters.get_bool("use_mean_log_mass", False):
        average_properties |= ClusterProperty.MASS

    survey_name = "numcosmo_simulated_redshift_richness"
    likelihood = ConstGaussian(
        [BinnedClusterNumberCounts(average_properties, survey_name, integrator)]
    )

    # Read in sacc data
    sacc_file_nm = "cluster_redshift_richness_sacc_data.fits"
    sacc_path = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/cluster_number_counts/")
    )
    sacc_data = sacc.Sacc.load_fits(os.path.join(sacc_path, sacc_file_nm))
    likelihood.read(sacc_data)

    cluster_abundance = get_cluster_abundance()
    modeling_tools = ModelingTools(cluster_abundance=cluster_abundance)

    return likelihood, modeling_tools
