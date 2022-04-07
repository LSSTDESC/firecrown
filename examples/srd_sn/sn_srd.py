#!/usr/bin/env python

from pprint import pprint
import os
import firecrown
import firecrown.likelihood.gauss_family.statistic.supernova as SN
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian

import sacc

# Sources

params = set()

sources = {}

params.add("ia_bias")

snia_stats = SN.Supernova(sacc_tracer="sn_ddf_sample")

# Likelihood

lk = ConstGaussian(statistics=[snia_stats])

# SACC file

saccfile = os.path.expanduser(
    os.path.expandvars("${FIRECROWN_EXAMPLES_DIR}/srd_sn/srd-y1-converted.sacc")
)
sacc_data = sacc.Sacc.load_fits(saccfile)

lk.read(sacc_data)
lk.set_params_names(params)

# Final object

likelihood = lk
