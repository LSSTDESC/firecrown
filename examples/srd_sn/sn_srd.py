#!/usr/bin/env python

from pprint import pprint
import os
import firecrown
import firecrown.ccl.two_point
from firecrown.ccl.sources import *
from firecrown.ccl.statistics import *
from firecrown.ccl.systematics import *
from firecrown.ccl.likelihoods import *
import firecrown.ccl.sources.wl_source
import sacc

# Sources

params = set([])

sources = {}

params.add("ia_bias")

stats = {"snia_theory": Supernova(sacc_tracer = "sn_ddf_sample")}

# Likelihood

lk = ConstGaussianLogLike(data_vector=list(stats.keys()))

# SACC file

saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_EXAMPLES_DIR}/srd_sn/srd-y1-converted.sacc"
    )
)
sacc_data = sacc.Sacc.load_fits(saccfile)

for name, stat in stats.items():
    stat.read(sacc_data)

lk.read(sacc_data, sources, stats)
print('AM****') #AM
lk.set_sources({})
lk.set_systematics({})
lk.set_statistics(stats)
lk.set_params_names(params)

# Final object

likelihood = lk
