"""Define the likelihood factory function for the cosmic shear example.
"""

import os

import sacc

import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian


def build_likelihood(_):
    """Create a firecrown likelihood for a cosmic shear analysis."""
    # Creating sources, each one maps to a specific section of a SACC file. In
    # this case trc0, trc1 describe weak-lensing probes. The sources are saved
    # in a dictionary since they will be used by one or more two-point
    # functions.

    # We include a photo-z shift bias (a constant shift in dndz). We also
    # have a different parameter for each bin, so here again we use the
    # src{i}_ prefix.
    source0 = wl.WeakLensing(
        sacc_tracer="trc0", systematics=[wl.PhotoZShift(sacc_tracer="trc0")]
    )
    source1 = wl.WeakLensing(
        sacc_tracer="trc1", systematics=[wl.PhotoZShift(sacc_tracer="trc1")]
    )

    # Now that we have all sources we can instantiate all the two-point
    # functions. For each one we create a new two-point function object.

    # Creating all auto/cross-correlations two-point function objects for
    # the weak-lensing probes.
    stats = [
        TwoPoint("galaxy_shear_cl_ee", source0, source0),
        TwoPoint("galaxy_shear_cl_ee", source0, source1),
        TwoPoint("galaxy_shear_cl_ee", source1, source1),
    ]

    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.
    likelihood = ConstGaussian(statistics=stats)

    # We load the correct SACC file.
    saccfile = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/cosmicshear/cosmicshear.fits")
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # two-point functions will receive the appropriated sections of the SACC
    # file and the sources their respective dndz.
    likelihood.read(sacc_data)

    # This script will be loaded by the appropriate connector. The framework
    # will call the factory function that should return a Likelihood instance.
    return likelihood
