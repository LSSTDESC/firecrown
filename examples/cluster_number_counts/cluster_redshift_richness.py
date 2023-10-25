"""Likelihood factory function for cluster number counts."""

import os

import pyccl as ccl
import sacc

from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.integrator.scipy_integrator import ScipyIntegrator
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.statistic.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.models.cluster.abundance_data import AbundanceData
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.kernel import (
    Completeness,
    DESY1PhotometricRedshift,
    Purity,
    SpectroscopicRedshift,
)
from firecrown.models.cluster.mass_proxy.murata import MurataBinned


def get_cluster_abundance(sky_area):
    hmf = ccl.halos.MassFuncBocquet16()
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_abundance = ClusterAbundance(
        min_mass, max_mass, min_z, max_z, hmf, sky_area
    )

    # Create and add the kernels you want in your cluster abundance
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    mass_observable_kernel = MurataBinned(pivot_mass, pivot_redshift)
    cluster_abundance.add_kernel(mass_observable_kernel)

    # redshift_proxy_kernel = SpectroscopicRedshift()
    redshift_proxy_kernel = DESY1PhotometricRedshift()
    cluster_abundance.add_kernel(redshift_proxy_kernel)

    # completeness_kernel = Completeness()
    # cluster_abundance.add_kernel(completeness_kernel)

    purity_kernel = Purity()
    cluster_abundance.add_kernel(purity_kernel)

    return cluster_abundance


def build_likelihood(build_parameters):
    """
    Here we instantiate the number density (or mass function) object.
    """
    integrator = NumCosmoIntegrator()
    # integrator = ScipyIntegrator()

    # Pull params for the likelihood from build_parameters
    use_cluster_counts = build_parameters.get_bool("use_cluster_counts", True)
    use_mean_log_mass = build_parameters.get_bool("use_mean_log_mass", False)
    survey_name = "numcosmo_simulated_redshift_richness"
    likelihood = ConstGaussian(
        [
            BinnedClusterNumberCounts(
                use_cluster_counts, use_mean_log_mass, survey_name, integrator
            )
        ]
    )

    # Read in sacc data
    sacc_file_nm = "cluster_redshift_richness_sacc_data.fits"
    sacc_path = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/cluster_number_counts/")
    )
    sacc_data = sacc.Sacc.load_fits(os.path.join(sacc_path, sacc_file_nm))
    likelihood.read(sacc_data)

    sacc_adapter = AbundanceData(
        sacc_data, survey_name, use_cluster_counts, use_mean_log_mass
    )
    cluster_abundance = get_cluster_abundance(sacc_adapter.survey_tracer.sky_area)
    modeling_tools = ModelingTools(cluster_abundance=cluster_abundance)

    return likelihood, modeling_tools
