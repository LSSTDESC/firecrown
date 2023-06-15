import os
import sacc
import firecrown.likelihood.gauss_family.statistic.supernova as sn
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
<<<<<<< HEAD

=======
import sacc
import sys
from typing import Dict
>>>>>>> 51c545aec15706df1776d47169639b5e57a6745a

def build_likelihood(_):
    """Build the Firecrown likelihood object. We have no extra tools for this
    example."""
    # Here we instantiate the necessary statistic object to deal with SNIa data.
    snia_stats = sn.Supernova(sacc_tracer="sn_ddf_sample")

<<<<<<< HEAD
    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.
    lk = ConstGaussian(statistics=[snia_stats])
=======
sources = {}  # type: Dict
>>>>>>> 51c545aec15706df1776d47169639b5e57a6745a

    #    We load the correct SACC file.
    saccfile = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/srd_sn/srd-y1-converted.sacc")
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # The read likelihood method is called passing the loaded SACC file, the
    # statistic functions will receive the appropriated sections of the SACC
    # file..
    lk.read(sacc_data)

<<<<<<< HEAD
    # This script will be loaded by the appropriated connector. The framework
    # will call the factory function that should return a Likelihood instance.
    return lk
=======
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
>>>>>>> 51c545aec15706df1776d47169639b5e57a6745a
