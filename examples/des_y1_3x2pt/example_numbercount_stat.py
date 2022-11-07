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


sac_file = sacc.Sacc.load_fits("des_y1_3x2pt_sacc_data.fits")
#print(sac_file.data)
stats = NumberCountStat("counts", f"src{1}")
rp = stats.required_parameters()
lista_t = [stats]
lk = ConstGaussian(lista_t)
lk.read(sac_file)

#print(sacc.standard_types)
#print(sac_file.data)