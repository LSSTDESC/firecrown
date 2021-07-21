import math
import numpy as np

from pprint import pprint

import firecrown

from cobaya.likelihood import Likelihood


class LikelihoodConnector(Likelihood):
    """
    A class implementing cobaya.likelihood.Likelihood ...

    ...

    Attributes
    ----------
    ... : str
        ...

    Methods
    -------
    ...(...)
        ....
    """

    firecrownIni = None

    def initialize(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        self.config, self.data = firecrown.parse(self.firecrownIni)

    def get_param(self, p):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return self._current_state["derived"][p]

    def initialize_with_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def initialize_with_provider(self, provider):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        self.provider = provider

    def get_can_provide_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return []

    def get_allow_agnostic(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return False

    def get_requirements(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        likelihood_requires = {"ccl": None}
        return likelihood_requires

    def must_provide(self, **requirements):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def logp(self, **params_values):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        ccl = self.provider.get_ccl()
        loglikes, _, _, _, _, _ = firecrown.compute_loglike(cosmo=ccl, data=self.data)
        return np.sum([v for v in loglikes.values() if v is not None])

