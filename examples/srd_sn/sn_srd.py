#!/usr/bin/env python

import os
import firecrown.likelihood.gauss_family.statistic.supernova as sn
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
import sacc
import sys

# Sources

sources = {}

snia_stats = sn.Supernova(sacc_tracer="sn_ddf_sample")

# Likelihood

lk = ConstGaussian(statistics=[snia_stats])

# SACC file
if len(sys.argv) == 1:
    saccfile = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/srd_sn/srd-y1-converted.sacc")
    )
else:
    file = sys.argv[1]  # Input sacc file name
    saccfile = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/srd_sn/" + file)
    )
sacc_data = sacc.Sacc.load_fits(saccfile)

lk.read(sacc_data)

# Final object

likelihood = lk
