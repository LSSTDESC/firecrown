#!/usr/bin/env python
import os
import cProfile as profile
import pstats
import numpy as np
import pyccl as ccl
import sacc
from firecrown.likelihood.gauss_family.statistic.number_counts_stat import *

from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
sacc.__version__
import torchquad
import six
torchquad._deployment_test()
cosmo = ccl.Cosmology(Omega_c=0.22, Omega_b=0.0448,
                      h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04)
print(cosmo)
#sac_file = sacc.Sacc.load_fits("./clusters.sacc")
sac_file = sacc.Sacc.load_fits("./clusters_real_data.sacc")
stats = NumberCountStat('cluster_counts_true_mass', 'cluster_mass_count_wl', massfunc = ['bocquet16', True])
#stats2 = NumberCountStat('cluster_counts_true_mass', 'cluster_mass_count_wl', massfunc = ['tinker08'])
lista_t = [stats]
#lista_t2 = [stats2]
lk = ConstGaussian(lista_t)
lk.read(sac_file)
lk.reset()
lk.get_derived_parameters()
prof = profile.Profile()
prof.enable()

log  = lk.compute_loglike(cosmo)
print(log)
#lk2 = ConstGaussian(lista_t2)
#lk2.read(sac_file)
#lk2.reset()
#lk2.get_derived_parameters()

#log2  = lk2.compute_loglike(cosmo)
#print(log2)
prof.disable()
stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
stats.print_stats(20)
#print(sacc.standard_types)
#print(sac_file.data)
