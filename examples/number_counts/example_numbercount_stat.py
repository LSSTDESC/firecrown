#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyccl as ccl
import sacc
from firecrown.likelihood.gauss_family.statistic.number_counts_stat import *

from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
sacc.__version__

cosmo = ccl.Cosmology(Omega_c=0.22, Omega_b=0.0448,
                      h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04)
print(cosmo)
#sac_file = sacc.Sacc.load_fits("./clusters.sacc")
sac_file = sacc.Sacc.load_fits("./clusters_real_data.sacc")
stats = NumberCountStat('cluster_counts_true_mass', 'cluster_mass_count_wl')
rp = stats.required_parameters()
lista_t = [stats]
lk = ConstGaussian(lista_t)
lk.read(sac_file)
lk.reset()
lk.get_derived_parameters()
print(lk.compute_loglike(cosmo))
#print(sacc.standard_types)
#print(sac_file.data)
plt.figure(figsize=(10,6))
plt.plot(1,1)