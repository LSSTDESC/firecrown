"""Cobaya Likelihood Connector.

Module for providing a likelihood for use in Cobaya.

This module provides the class :class:`LikelihoodConnector`, which is an implementation
of a Cobaya likelihood.
"""

import numpy as np
import numpy.typing as npt
from cobaya.likelihood import Likelihood

from firecrown.likelihood.likelihood import load_likelihood, NamedParameters
from firecrown.parameters import ParamsMap


class LikelihoodConnector(Likelihood):
    """A class implementing cobaya.likelihood.Likelihood."""

    likelihood: Likelihood
    firecrownIni: str
    derived_parameters: list[str] = []
    build_parameters: NamedParameters

    def initialize(self):
        """Initialize the likelihood object by loading its Firecrown configuration."""
        if not hasattr(self, "build_parameters"):
            build_parameters = NamedParameters()
        else:
            if isinstance(self.build_parameters, dict):
                build_parameters = NamedParameters(self.build_parameters)
            else:
                if not isinstance(self.build_parameters, NamedParameters):
                    raise TypeError(
                        "build_parameters must be a NamedParameters or dict"
                    )
                build_parameters = self.build_parameters

        self.likelihood, self.tools = load_likelihood(
            self.firecrownIni, build_parameters
        )

    def initialize_with_params(self) -> None:
        """Complete the initialization of a LikelihoodConnector object.

        Required by Cobaya.

        This version has nothing to do.
        """

    def initialize_with_provider(self, provider) -> None:
        """Set the obejct's provider.

        Required by Cobaya.

        :param provider: A Cobaya provider.
        """
        self.provider = provider

    def get_can_provide_params(self) -> list[str]:
        """Return the list of params provided.

        Required by Cobaya.

        Returns an empty list.
        """
        return self.derived_parameters

    def get_allow_agnostic(self) -> bool:
        """Is it allowed to pass all unassigned input parameters to this component.

        Required by Cobaya.

        Return False.
        """
        return False

    def get_requirements(
        self,
    ) -> dict[str, None | dict[str, npt.NDArray[np.float64]] | dict[str, object]]:
        """Returns a dictionary.

        Returns a dictionary with keys corresponding the contained likelihood's
        required parameter, plus "pyccl". All values are None.

        Required by Cobaya.
        :return: a dictionary
        """
        likelihood_requires: dict[
            str, None | dict[str, npt.NDArray[np.float64]] | dict[str, object]
        ] = {"pyccl": None}
        required_params = (
            self.likelihood.required_parameters() + self.tools.required_parameters()
        )

        for param_name in required_params.get_params_names():
            likelihood_requires[param_name] = None

        return likelihood_requires

    def must_provide(self, **requirements) -> None:
        """Required by Cobaya.

        This version does nothing.
        """

    def logp(self, **params_values) -> float:
        """Return the log of the calculated likelihood.

        Required by Cobaya.
        :params values: The values of the parameters to use.
        """
        pyccl = self.provider.get_pyccl()

        self.likelihood.update(ParamsMap(params_values))
        self.tools.update(ParamsMap(params_values))
        self.tools.prepare(pyccl)

        loglike = self.likelihood.compute_loglike(self.tools)

        derived_params_collection = self.likelihood.get_derived_parameters()
        assert derived_params_collection is not None
        for section, name, val in derived_params_collection:
            params_values["_derived"][f"{section}__{name}"] = val

        self.likelihood.reset()
        self.tools.reset()

        return loglike
