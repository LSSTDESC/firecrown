"""Likelihood factory function for cluster number counts."""

import os
from typing import Any, Dict

import pyccl as ccl
import sacc

from firecrown.likelihood.gauss_family.statistic.cluster_number_counts import (
    ClusterNumberCounts,
)
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster_abundance import ClusterAbundance
from firecrown.models.mass_observable import MassObservableFactory, MassObservableType
from firecrown.models.redshift import RedshiftFactory, RedshiftType
from firecrown.parameters import ParamsMap


def build_likelihood(build_parameters):
    """
    Here we instantiate the number density (or mass function) object.
    """

    stats = ClusterNumberCounts(survey_name="numcosmo_simulated_redshift_richness")
    stats.use_cluster_counts = build_parameters.get_bool("use_cluster_counts", True)
    stats.use_mean_log_mass = build_parameters.get_bool("use_mean_log_mass", False)
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
    hmf = ccl.halos.MassFuncTinker08()
    cluster_abundance = ClusterAbundance(hmf)

    mass_observable = MassObservableFactory.create(
        MassObservableType.MU_SIGMA,
        ParamsMap(
            {
                "pivot_mass": 14.625862906,
                "pivot_redshift": 0.6,
                "min_mass": 13.0,
                "max_mass": 15.0,
            }
        ),
    )
    cluster_abundance.add_kernel(mass_observable)

    redshift = RedshiftFactory.create(RedshiftType.SPEC)
    cluster_abundance.add_kernel(redshift)

    modeling = ModelingTools(cluster_abundance=cluster_abundance)

    return lk, modeling
