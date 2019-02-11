import numpy as np

import pyccl as ccl
from firecrown.ccl.core import Systematic

# constant from KEB16, near eqn 7
C1RHOC = 0.0134


class DESNLASystematic(Systematic):
    """KEB NLA systematic.

    This systematic adds the KEB non-linear, linear alignment (NLA) intrinsic
    alignment model which varies with redshift, luminosity, and
    the growth function.

    Parameters
    ----------
    eta_ia : str
        The mame of redshift dependence parameter of the intrinsic alignment
        signal.
    Omega_b : str
        The name of the parameter for the baryon density at z = 0.
    Omega_c : str
        The name of the patameter for the cold dark matter density at z = 0.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self, eta_ia, Omega_b, Omega_c):
        self.eta_ia = eta_ia
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c

        # set internal **constants**
        self._zpiv_eta_ia = 0.62

    def apply(self, cosmo, params, source):
        """Apply a linear alignment systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        ia_bias = (
            (params[self.Omega_b] + params[self.Omega_c]) *
            C1RHOC /
            ccl.growth_factor(cosmo, 1.0 / (1.0 + source.z_)) *
            np.power((1 + source.z_) / (1 + self._zpiv_eta_ia),
                     params[self.eta_ia]))

        source.ia_bias_ *= np.array(ia_bias)
