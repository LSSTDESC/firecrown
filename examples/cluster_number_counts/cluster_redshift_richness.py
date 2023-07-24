"""Likelihood factory function for cluster number counts."""

import os
from typing import Any, Dict

import pyccl as ccl

from firecrown.likelihood.gauss_family.statistic.cluster_number_counts import (
    ClusterNumberCounts,
)
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster_abundance import ClusterAbundance
from firecrown.models.cluster_mass_rich_proxy import ClusterMassRich
from firecrown.models.cluster_redshift_spec import ClusterRedshiftSpec
from firecrown.sacc_support import sacc


def build_likelihood(build_parameters):
    """
    Here we instantiate the number density (or mass function) object.
    """

    pivot_mass = 14.625862906
    pivot_redshift = 0.6

    cluster_mass_r = ClusterMassRich(pivot_mass, pivot_redshift)
    cluster_z = ClusterRedshiftSpec()

    hmd_200 = ccl.halos.MassDef200m()
    hmf_args: Dict[str, Any] = {}
    hmf_name = "Tinker08"

    cluster_abundance = ClusterAbundance(hmd_200, hmf_name, hmf_args)

    stats = ClusterNumberCounts(
        "numcosmo_simulated_redshift_richness",
        cluster_abundance,
        cluster_mass_r,
        cluster_z,
        use_cluster_counts=build_parameters.get_bool("use_cluster_counts", True),
        use_mean_log_mass=build_parameters.get_bool("use_mean_log_mass", False),
    )
    stats_list = [stats]

    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.

    lk = ConstGaussian(stats_list)

    # We load the correct SACC file.
    saccfile = os.path.expanduser(
        os.path.expandvars(
            "${FIRECROWN_DIR}/examples/cluster_number_counts/"
            "cluster_redshift_richness_sacc_data.fits"
        )
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # The read likelihood method is called passing the loaded SACC file, the
    # cluster number count functions will receive the appropriated sections
    # of the SACC file.
    lk.read(sacc_data)

    # This script will be loaded by the appropriated connector. The framework
    # then looks for the `likelihood` variable to find the instance that will
    # be used to compute the likelihood.
    modeling = ModelingTools()

    return lk, modeling
