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


sac_file = sacc.Sacc.load_fits("./clusters.sacc")
sac_file2 = sacc.Sacc.load_fits("/home/eduardo/firecrown/examples/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits")
#print(sac_file.data)
#print(sac_file2.data)
stats = NumberCountStat("cluster_mass_count_wl")
rp = stats.required_parameters()
lista_t = [stats]
lk = ConstGaussian(lista_t)
lk.read(sac_file)

#print(sacc.standard_types)
#print(sac_file.data)