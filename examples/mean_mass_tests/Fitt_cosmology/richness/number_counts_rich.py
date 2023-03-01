#!/usr/bin/env python

import os

from firecrown.likelihood.gauss_family.statistic.cluster_number_counts import (
    ClusterNumberCounts,
)
from firecrown.models.ccl_density import CCLDensity
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
import sacc


def build_likelihood(_):
    """
    Here we instantiate the number density (or mass function) object.
    """
    massfunc = CCLDensity("mean", "Tinker08")
    stats = ClusterNumberCounts(
        "cluster_counts_richness_proxy", "cluster_mass_count_wl", massfunc
    )
    stats = [stats]
    """
        Here we instantiate the actual likelihood. The statistics argument carry
        the order of the data/theory vector.
    """
    lk = ConstGaussian(stats)

    """
        We load the correct SACC file.
    """
    saccfile = os.path.expanduser(
        os.path.expandvars(
            "${FIRECROWN_DIR}/examples/mean_mass_tests/Fitt_cosmology/richness/clusters_numcosmo_richness_data.sacc"
        )
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)

    """
        The read likelihood method is called passing the loaded SACC file, the
        cluster number count functions will receive the appropriated sections
        of the SACC file.
    """
    lk.read(sacc_data)

    """
        This script will be loaded by the appropriated connector. The framework
        then looks for the `likelihood` variable to find the instance that will
        be used to compute the likelihood.
    """
    modeling = ModelingTools()
    return lk, modeling
