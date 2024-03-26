"""Cobaya Likelihood Connector.

Module for providing a likelihood for use in Cobaya.

This module provides the class :class:`LikelihoodConnector`, which is an implementation
of a Cobaya likelihood.
"""

from typing import Union

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
        """Required by Cobaya.

        This version has nothing to do.
        """

    def initialize_with_provider(self, provider) -> None:
        """Required by Cobaya.

        Sets instance's provided to the given provider.
        """
        self.provider = provider

    def get_can_provide_params(self) -> list[str]:
        """Required by Cobaya.

        Returns an empty list.
        """
        return self.derived_parameters

    def get_allow_agnostic(self):
        """Required by Cobaya.

        Return False.
        """
        return False

    def get_requirements(
        self,
    ) -> dict[str, Union[None, dict[str, npt.NDArray[np.float64]], dict[str, object]]]:
        """Required by Cobaya.

        Returns a dictionary with keys corresponding the contained likelihood's
        required parameter, plus "pyccl". All values are None.
        """
        likelihood_requires: dict[
            str, Union[None, dict[str, npt.NDArray[np.float64]], dict[str, object]]
        ] = {"pyccl": None}
        required_params = (
            self.likelihood.required_parameters() + self.tools.required_parameters()
        )

        for param_name in required_params.get_params_names():
            likelihood_requires[param_name] = None

        return likelihood_requires

    def must_provide(self, **requirements):
        """Required by Cobaya.

        This version does nothing.
        """

    def logp(self, **params_values) -> float:
        """Required by Cobaya.

        Return the (log) calculated likelihood.
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
