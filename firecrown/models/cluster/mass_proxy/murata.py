from typing import List, Tuple

import numpy as np

from firecrown.updatable import Updatable
from firecrown import parameters

from firecrown.models.cluster_theory.mass_proxy import murata as theo_murata

from firecrown.models.cluster.kernel import KernelType
from firecrown.models.cluster.mass_proxy.gaussian import MassRichnessGaussian


def _set_parameters(self):
    # Updatable parameters
    self.pars = Updatable()
    self.pars.mu_p0 = parameters.create()
    self.pars.mu_p1 = parameters.create()
    self.pars.mu_p2 = parameters.create()
    self.pars.sigma_p0 = parameters.create()
    self.pars.sigma_p1 = parameters.create()
    self.pars.sigma_p2 = parameters.create()


### used to be MassRichnessMuSigma ###
class MurataBinned(theo_murata.MurataBinned):
    pass


class MurataUnbinned(theo_murata.MurataUnbinned):
    pass


MurataBinned.set_parameters = _set_parameters
MurataUnbinned.set_parameters = _set_parameters
