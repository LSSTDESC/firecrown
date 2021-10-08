#!/usr/bin/env python

from pprint import pprint
import os
import firecrown

import firecrown.ccl.two_point

from firecrown.ccl.sources import *
from firecrown.ccl.statistics import *
from firecrown.ccl.systematics import *
from firecrown.ccl.likelihoods import *

import sacc

# Sources

params = set([])

sources = {}

params.add ("A0")
for i in range(4):
    systematics_params = [f'delta_z_src{i}', 'des_ia', f'shear_bias_src{i}']
    sources[f'src{i}'] = WLSource (sacc_tracer=f"src{i}", 
                                   ia_bias="A0",
                                   systematics=systematics_params)
                   
for i in range(5):
    sources[f'lens{i}'] = NumberCountsSource (sacc_tracer=f'lens{i}',
                                              bias=f'bias_lens{i}',
                                              systematics=[f'delta_z_lens{i}'])
    params.add (f'bias_lens{i}')

# Statistics

stats = {}
for stat, sacc_stat in [('xip', 'galaxy_shear_xi_plus'), ('xim', 'galaxy_shear_xi_minus')]:
    for i in range(4):
        for j in range(i, 4):
            stats[f'{stat}_src{i}_src{j}'] = TwoPointStatistic (sources = [f'src{i}', f'src{j}'], 
                                                                sacc_data_type = sacc_stat)
for j in range(5):
    for i in range(4):
        stats[f'gammat_lens{j}_src{i}'] = TwoPointStatistic (sources = [f'lens{j}', f'src{i}'],
                                                             sacc_data_type = 'galaxy_shearDensity_xi_t')


for i in range(5):
    stats[f'wtheta_lens{i}_lens{i}'] = TwoPointStatistic (sources = [f'lens{i}', f'lens{i}'],
                                                          sacc_data_type = 'galaxy_density_xi')

# Systematics

systematics = {}

for i in range(5):
    systematics[f'delta_z_lens{i}'] = PhotoZShiftBias (delta_z = f'lens{i}_delta_z')
    params.add (f'lens{i}_delta_z')

for i in range(4):
    systematics[f'delta_z_src{i}'] = PhotoZShiftBias (delta_z = f'src{i}_delta_z')
    params.add (f'src{i}_delta_z')

params.add ('eta_ia')
params.add ('alphag_ia')
params.add ('z_piv_ia')
systematics['des_ia'] = LinearAlignmentSystematic (alphaz='eta_ia', 
                                                   alphag='alphag_ia',
                                                   z_piv='z_piv_ia')
for i in range(4):
    systematics[f'shear_bias_src{i}'] = MultiplicativeShearBias (m=f'src{i}_mult_bias')
    params.add (f'src{i}_mult_bias')

# Likelihood

lk = ConstGaussianLogLike (data_vector = list (stats.keys ()))

# SACC file

saccfile = os.path.expanduser(os.path.expandvars('${FIRECROWN_EXAMPLES_DIR}/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits'))
sacc_data = sacc.Sacc.load_fits(saccfile)

for name, source in sources.items():
    source.read (sacc_data)

for name, stat in stats.items():
    stat.read (sacc_data, sources)

lk.read (sacc_data, sources, stats)

lk.set_sources (sources)
lk.set_systematics (systematics)
lk.set_statistics (stats)
lk.set_params_names (params)

# Final object

likelihood = lk


