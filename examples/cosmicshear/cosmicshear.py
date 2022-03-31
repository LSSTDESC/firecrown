#!/usr/bin/env python

from pprint import pprint
import os
import firecrown

from firecrown.likelihood.gauss_family.statistic.source.weak_lensing import WeakLensing
from firecrown.likelihood.gauss_family.statistic.source.weak_lensing import PhotoZShift as WLPhotoZShift

from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian

import sacc

# Sources

params = set()

'''
    Creating sources, each one maps to a specific section of a SACC file. In 
    this case trc0, trc1 describe weak-lensing probes. The sources are saved 
    in a dictionary since they will be used by one or more two-point function.
'''
sources = {}

for i in range(2):
    '''
        We include a photo-z shift bias (a constant shift in dndz). We also 
        have a different parameter for each bin, so here again we use the
        src{i}_ prefix. 
    '''
    pzshift = WLPhotoZShift(sacc_tracer=f"trc{i}")
    params.add(f"trc{i}_delta_z")

    '''
        Now we can finally create the weak-lensing source that will compute the
        theoretical prediction for that section of the data, given the 
        systematics.
    '''
    sources[f"trc{i}"] = WeakLensing(
        sacc_tracer=f"trc{i}", systematics=[pzshift]
    )

'''
    Now that we have all sources we can instantiate all the two-point 
    functions. For each one we create a new two-point function object.
'''
stats = {}

'''
    Creating all auto/cross-correlations two-point function objects for 
    the weak-lensing probes.
'''
for i in range(2):
    for j in range(i, 2):
        stats[f"trc{i}_trc{j}"] = TwoPoint(
            source0=sources[f"trc{i}"], source1=sources[f"trc{j}"], 
            sacc_data_type="galaxy_shear_cl_ee"
        )

'''
    Here we instantiate the actual likelihood. The statistics argument carry
    the order of the data/theory vector.
'''
lk = ConstGaussian(statistics=list(stats.values()))

'''
    We load the correct SACC file.
'''
saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_EXAMPLES_DIR}/cosmicshear/cosmicshear.fits"
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
