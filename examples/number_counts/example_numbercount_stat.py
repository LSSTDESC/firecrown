#!/usr/bin/env python

import pyccl as ccl
import sacc
from typing import List

from firecrown.likelihood.gauss_family.statistic.number_counts_stat import (
    NumberCountStat,
)
from firecrown.models.ccl_density import CCLDensity
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.statistic.statistic import Statistic

"""
In here we define the cosmology to be used to compute the statistics.
"""
cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)

"""
This is the sacc file to be used. It contains a simulation of
clusters with their mass and redshift.
sac_file = sacc.Sacc.load_fits("./clusters_simulated_data.sacc")
"""
sac_file = sacc.Sacc.load_fits("./clusters_simulated_richness_data.sacc")
"""
We then initiate the massfunction object. It can be a critical mass function
or mean mass funcion. Some mass functions have the possibility of using
dark matter + baryonic matter to compute the cluster mass, which is set by
the True entry in the function below. This is optional.
Otherwise, only dark matter is used as default.
"""
massfunc = CCLDensity("critical", "Bocquet16", False)

"""
Initiate the statics object
"""
stats = NumberCountStat(
    "cluster_counts_richness_proxy", "cluster_mass_count_wl", massfunc
)
list_stats: List[Statistic] = [stats]

"""
Initiate the likelihood object, which will read the sacc file and then
 compute the log(Likelihood).
"""
lk = ConstGaussian(list_stats)
lk.read(sac_file)
log = lk.compute_loglike(cosmo)
print(f"The log(L) is: {log}")
