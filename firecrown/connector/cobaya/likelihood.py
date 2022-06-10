"""Module for providing a likelihood for use in Cobaya.

This module provides the class LikelihoodConnector, which is an implementation
of a Cobaya likelihood.
"""
from typing import List, Dict

from cobaya.likelihood import Likelihood

from firecrown.likelihood.likelihood import load_likelihood
from firecrown.parameters import ParamsMap


class LikelihoodConnector(Likelihood):
    """A class implementing cobaya.likelihood.Likelihood."""

    firecrownIni = None

    def initialize(self):
        """Initialize the likelihood object by loading its Firecrown
        configuration."""
        self.likelihood = load_likelihood(self.firecrownIni)

    def get_param(self, p: str):
        """Return the current value of the parameter named 'p'."""
        return self._current_state["derived"][p]

    def initialize_with_params(self) -> None:
        """Required by Cobaya.

        This version has nothing to do.
        """

    def initialize_with_provider(self, provider) -> None:
        """Required by Cobaya.

        Sets instance's provided to the given provider.
        """

        self.provider = provider

    def get_can_provide_params(self) -> List:
        """Required by Cobaya.

        Returns an empty list.
        """
        return []

    def get_allow_agnostic(self):
        """Required by Cobaya.

        Return False.
        """
        return False

    def get_requirements(self) -> Dict:
        """Required by Cobaya.

        Returns a dictionary with keys corresponding the contained likelihood's
        required parameter, plus "ccl". All values are None.
        """
        likelihood_requires = {"ccl": None}

        required_params = self.likelihood.required_parameters()

        for param_name in required_params.get_params_names():
            likelihood_requires[param_name] = None

        return likelihood_requires

    def must_provide(self, **requirements):
        """Required by Cobaya.

        This version does nothing.
        """

    def logp(self, **params_values) -> float:
        """Requried by Cobaya.

        Return the (log) calculated likelihood.
        """
        ccl = self.provider.get_ccl()
        # loglikes, _, _, _, _, _ = firecrown.compute_loglike(cosmo=ccl, data=self.data)
        # return np.sum([v for v in loglikes.values() if v is not None])

        self.likelihood.update(ParamsMap(params_values))

        return self.likelihood.compute_loglike(ccl, ParamsMap(params_values))
