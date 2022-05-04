#!/usr/bin/env python

from pprint import pprint
import os
import firecrown
#import firecrown.ccl.two_point
from firecrown.ccl.sources import *
from firecrown.ccl.statistics import *
from firecrown.ccl.systematics import *
from firecrown.ccl.likelihoods import *
#import firecrown.ccl.sources.wl_source

import sacc

# Sources
# Systematics
params = set()

sources = {}

params.add("delta_z") # redshift bias

snia_stats = Supernova(sacc_tracer = "sn_ddf_sample")

# Likelihood

lk = ConstGaussianLogLike(statistics=[snia_stats])

# SACC file

saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_EXAMPLES_DIR}/srd_sn/srd-y1-converted.sacc"
    )
)
sacc_data = sacc.Sacc.load_fits(saccfile)

lk.read(sacc_data)
lk.set_params_names(params)

# Final object

likelihood = lk
