"""Cluster Number Counts Likelihood Example.
"""

import os

import sacc
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.gauss_family.statistic.cluster_number_counts import (
    ClusterNumberCounts,
)
from firecrown.models.ccl_density import CCLDensity
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.models.cluster_mass import ClusterMass
from firecrown.models.cluster_redshift import ClusterRedshift
import pyccl as ccl

def build_likelihood(_):
    """
    Here we instantiate the number density (or mass function) object.
    """
    hmd_200 = ccl.halos.MassDef200c()
    hmf_args = [True, True]
    hmf_200 = ccl.halos.MassFuncBocquet16
    cluster_m = ClusterMass(hmd_200, hmf_200, hmf_args=hmf_args)
    cluster_z = ClusterRedshift()


    stats = ClusterNumberCounts(
        "cluster_counts_true_mass", "cluster_mass_count_wl", cluster_m, cluster_z
    )
    stats = [stats]

    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.
    lk = ConstGaussian(stats)  # pylint: disable=invalid-name

    # We load the correct SACC file.
    saccfile = os.path.expanduser(
        os.path.expandvars(
            "${FIRECROWN_DIR}/examples/cluster_number_counts/"
            + "clusters_simulated_data.sacc"
        )
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)
    modeling_tools = ModelingTools()
    # The read likelihood method is called passing the loaded SACC file, the
    # cluster number count functions will receive the appropriated sections
    # of the SACC file.
    lk.read(sacc_data)
    # This script will be loaded by the appropriated connector. The framework
    # then looks for the `likelihood` variable to find the instance that will
    # be used to compute the likelihood.
    return lk, modeling_tools
