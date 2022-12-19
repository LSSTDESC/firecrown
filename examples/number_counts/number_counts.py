#!/usr/bin/env python

import os

from firecrown.likelihood.gauss_family.statistic.number_counts_stat import (
    NumberCountStat,
)
from firecrown.models.ccl_density import CCLDensity
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian

import sacc


def build_likelihood(_):

    massfunc = CCLDensity("critical", "Bocquet16", False)
    stats = NumberCountStat(
        "cluster_counts_true_mass", "cluster_mass_count_wl", massfunc
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
            "${FIRECROWN_DIR}/examples/number_counts/clusters_simulated_data.sacc"
        )
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)

    """
        The read likelihood method is called passing the loaded SACC file, the
        two-point functions will receive the appropriated sections of the SACC
        file and the sources their respective dndz.
    """
    lk.read(sacc_data)

    """
        This script will be loaded by the appropriated connector. The framework
        then looks for the `likelihood` variable to find the instance that will
        be used to compute the likelihood.
    """
    return lk
