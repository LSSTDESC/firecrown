#!/usr/bin/env python

import os
import firecrown.likelihood.gauss_family.statistic.supernova as sn
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian

import sacc

# Sources

params = set()

sources = {}

params.add("ia_bias")

snia_stats = sn.Supernova(sacc_tracer="sn_ddf_sample")

# Likelihood

lk = ConstGaussian(statistics=[snia_stats])

# SACC file

saccfile = os.path.expanduser(
    os.path.expandvars("${FIRECROWN_DIR}/examples/srd_sn/srd-y1-converted.sacc")
)
sacc_data = sacc.Sacc.load_fits(saccfile)

lk.read(sacc_data)
lk.set_params_names(list(params))

# Final object

likelihood = lk
