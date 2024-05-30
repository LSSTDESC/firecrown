"""Demonstration of the use of the :class:`Supernova` statistics object."""

import sacc
import firecrown.likelihood.gauss_family.supernova as sn
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.likelihood import NamedParameters


def build_likelihood(params: NamedParameters):
    """Build the Firecrown likelihood object.

    We have no extra tools for this example.
    """
    # Here we instantiate the necessary statistic object to deal with SNIa data.
    snia_stats = sn.Supernova(sacc_tracer="sn_ddf_sample")

    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.
    lk = ConstGaussian(statistics=[snia_stats])

    #    We load the correct SACC file.
    saccfile = params.get_string("sacc_file")
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # The read likelihood method is called passing the loaded SACC file, the
    # statistic functions will receive the appropriated sections of the SACC
    # file..
    lk.read(sacc_data)

    # This script will be loaded by the appropriated connector. The framework
    # will call the factory function that should return a Likelihood instance.
    return lk
