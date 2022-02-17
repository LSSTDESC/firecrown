#!/usr/bin/env python

from pprint import pprint
import os
import firecrown

import firecrown.ccl.two_point

from firecrown.ccl.sources import *
from firecrown.ccl.statistics import *
from firecrown.ccl.systematics import *
from firecrown.ccl.likelihoods import *

import firecrown.ccl.sources.wl_source as wl_source
import firecrown.ccl.sources.nc_source as nc_source

import sacc

# Sources

params = set([])

sources = {}

params.add("ia_bias")
params.add("alphaz")
params.add("alphag")
params.add("z_piv")

lai_systematic = wl_source.LinearAlignmentSystematic(
    sacc_tracer=""
)
for i in range(4):
    mbias = wl_source.MultiplicativeShearBias(
        sacc_tracer=f"src{i}"
    )
    params.add(f"src{i}_mult_bias")

    pzshift = wl_source.PhotoZShift(sacc_tracer=f"src{i}")
    params.add(f"src{i}_delta_z")

    sources[f"src{i}"] = WLSource(
        sacc_tracer=f"src{i}", systematics=[lai_systematic, mbias, pzshift]
    )

for i in range(5):
    
    pzshift = nc_source.PhotoZShift(sacc_tracer=f"lens{i}")
    
    sources[f"lens{i}"] = nc_source.NumberCountsSource(
        sacc_tracer=f"lens{i}", systematics=[pzshift]
    )
    params.add(f"lens{i}_bias")
    params.add(f"lens{i}_delta_z")

# Statistics

stats = {}
for stat, sacc_stat in [
    ("xip", "galaxy_shear_xi_plus"),
    ("xim", "galaxy_shear_xi_minus"),
]:
    for i in range(4):
        for j in range(i, 4):
            stats[f"{stat}_src{i}_src{j}"] = TwoPointStatistic(
                source0=sources[f"src{i}"], source1=sources[f"src{j}"], 
                sacc_data_type=sacc_stat
            )
for j in range(5):
    for i in range(4):
        stats[f"gammat_lens{j}_src{i}"] = TwoPointStatistic(
            source0=sources[f"lens{j}"], source1=sources[f"src{i}"], 
            sacc_data_type="galaxy_shearDensity_xi_t"
        )


for i in range(5):
    stats[f"wtheta_lens{i}_lens{i}"] = TwoPointStatistic(
        source0=sources[f"lens{i}"], source1=sources[f"lens{i}"], 
        sacc_data_type="galaxy_density_xi"
    )

# Likelihood
lk = ConstGaussianLogLike(statistics=list(stats.values()))

# SACC file
saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_EXAMPLES_DIR}/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits"
    )
)
sacc_data = sacc.Sacc.load_fits(saccfile)

lk.read(sacc_data)
lk.set_params_names(params)

# Final object

likelihood = lk
