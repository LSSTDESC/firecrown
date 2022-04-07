import math
import numpy as np

from pprint import pprint

import firecrown
from firecrown.likelihood.likelihood import load_likelihood
from firecrown.parameters import ParamsMap

from cobaya.likelihood import Likelihood

from firecrown.descriptors import TypeLikelihood


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

        self.likelihood = load_likelihood(self.firecrownIni)

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
        for param_name in self.likelihood.get_params_names():
            likelihood_requires[param_name] = None

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
        # loglikes, _, _, _, _, _ = firecrown.compute_loglike(cosmo=ccl, data=self.data)
        # return np.sum([v for v in loglikes.values() if v is not None])

        return self.likelihood.compute_loglike(ccl, ParamsMap(params_values))
