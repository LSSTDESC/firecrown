#!/usr/bin/env python

from pprint import pprint
import os
import firecrown

from firecrown.ccl.sources import *
from firecrown.ccl.statistics import *
from firecrown.ccl.systematics import *
from firecrown.ccl.likelihoods import *

import firecrown.ccl.sources.wl_source as wl_source
import firecrown.ccl.sources.nc_source as nc_source

import sacc

# Sources

params = set([])

'''
    Creating a systematic object that will be used by many sources. We need to 
    inform all parameter names (this will be automatic soon).
'''
params.add("ia_bias")
params.add("alphaz")
params.add("alphag")
params.add("z_piv")

lai_systematic = wl_source.LinearAlignmentSystematic(
    sacc_tracer=""
)

'''
    Creating sources, each one maps to a specific section of a SACC file. In 
    this case src0, src1, src2 and src3 describe weak-lensing probes. The 
    sources are saved in a dictionary since they will be used by one or more 
    two-point function.
'''
sources = {}

for i in range(4):
    '''
        Each weak-lensing section has its own multiplicative bias. Parameters 
        reflect this by using src{i}_ prefix. 
    '''
    mbias = wl_source.MultiplicativeShearBias(
        sacc_tracer=f"src{i}"
    )
    params.add(f"src{i}_mult_bias")

    '''
        We also include a photo-z shift bias (a constant shift in dndz). We 
        also have a different parameter for each bin, so here again we use the
        src{i}_ prefix. 
    '''
    pzshift = wl_source.PhotoZShift(sacc_tracer=f"src{i}")
    params.add(f"src{i}_delta_z")

    '''
        Now we can finally create the weak-lensing source that will compute the
        theoretical prediction for that section of the data, given the 
        systematics.
    '''
    sources[f"src{i}"] = WLSource(
        sacc_tracer=f"src{i}", systematics=[lai_systematic, mbias, pzshift]
    )


'''
    Creating the number counting sources. There are five sources each one 
    labeled by lens{i}. 
'''
for i in range(5):
    
    '''
        We also include a photo-z shift for the dndz.
    '''
    pzshift = nc_source.PhotoZShift(sacc_tracer=f"lens{i}")
        
    '''
        The source is created and saved (temporarely in the sources dict).
    '''
    sources[f"lens{i}"] = nc_source.NumberCountsSource(
        sacc_tracer=f"lens{i}", systematics=[pzshift]
    )
    params.add(f"lens{i}_bias")
    params.add(f"lens{i}_delta_z")

'''
    Now that we have all sources we can instantiate all the two-point 
    functions. The weak-lensing sources have two "data types", for each one we 
    create a new two-point function.
'''
stats = {}
for stat, sacc_stat in [
    ("xip", "galaxy_shear_xi_plus"),
    ("xim", "galaxy_shear_xi_minus"),
]:
    '''
        Creating all auto/cross-correlations two-point function objects for 
        the weak-lensing probes.
    '''
    for i in range(4):
        for j in range(i, 4):
            stats[f"{stat}_src{i}_src{j}"] = TwoPointStatistic(
                source0=sources[f"src{i}"], source1=sources[f"src{j}"], 
                sacc_data_type=sacc_stat
            )
    '''
        The following two-point function objects compute the cross correlations
        between the weak-lensing and number count sources.
    '''
for j in range(5):
    for i in range(4):
        stats[f"gammat_lens{j}_src{i}"] = TwoPointStatistic(
            source0=sources[f"lens{j}"], source1=sources[f"src{i}"], 
            sacc_data_type="galaxy_shearDensity_xi_t"
        )

    '''
        Finally the instances for the lensing auto-correlations are created.
    '''
for i in range(5):
    stats[f"wtheta_lens{i}_lens{i}"] = TwoPointStatistic(
        source0=sources[f"lens{i}"], source1=sources[f"lens{i}"], 
        sacc_data_type="galaxy_density_xi"
    )

'''
    Here we instantiate the actual likelihood. The statistics argument carry
    the order of the data/theory vector.
'''
lk = ConstGaussianLogLike(statistics=list(stats.values()))

'''
    We load the correct SACC file.
'''
saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_EXAMPLES_DIR}/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits"
    )
)
sacc_data = sacc.Sacc.load_fits(saccfile)

'''
    The read likelihood method is called passing the loaded SACC file, the 
    two-point functions will receive the appropriated sections of the SACC
    file and the sources their respective dndz. 
'''
lk.read(sacc_data)
lk.set_params_names(params)

'''
    This script will be loaded by the appropriated connector. The framework
    then looks for the `likelihood` variable to find the instance that will
    be used to compute the likelihood.
'''
likelihood = lk
