"""Likelihood of independent data sets as the sum of their log-likelihoods."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import sacc

from firecrown.likelihood_base import Likelihood
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import UpdatableCollection


class SumLikelihood(Likelihood):
    """Log-likelihood equal to the sum of the log-likelihoods of its components.

    The components are statistically independent likelihoods; the sum of their
    log-likelihoods is the log-likelihood of the joint data.

    The methods :meth:`update` and :meth:`reset` act on every component. Parameters
    shared by several components are required once and receive the same value.
    """

    def __init__(self, likelihoods: Iterable[Likelihood]) -> None:
        """Initialize the sum from its component likelihoods.

        :param likelihoods: the component likelihoods; at least one is required
        :raises ValueError: if no likelihood is given
        :raises TypeError: if a component is not a :class:`Likelihood`
        """
        components = list(likelihoods)
        if len(components) == 0:
            raise ValueError("SumLikelihood requires at least one likelihood.")
        for likelihood in components:
            if not isinstance(likelihood, Likelihood):
                raise TypeError(
                    f"Every component of a SumLikelihood must be a Firecrown "
                    f"`Likelihood`, received {type(likelihood)} instead."
                )
        # Unused parameters raise only if every component requests it.
        super().__init__(
            raise_on_unused_parameter=all(
                likelihood.raise_on_unused_parameter for likelihood in components
            )
        )
        self.likelihoods: UpdatableCollection[Likelihood] = UpdatableCollection(
            components
        )

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Call :meth:`read` of every component with the same SACC object.

        :param sacc_data: the SACC data object passed to every component
        """
        for likelihood in self.likelihoods:
            likelihood.read(sacc_data)

    def make_realization_vector(self) -> npt.NDArray[np.float64]:
        """Raise NotImplementedError; the components have separate data vectors.

        :raises NotImplementedError: always
        """
        raise NotImplementedError(
            "SumLikelihood has no single data vector; call make_realization_vector "
            "on each component in `likelihoods`."
        )

    def compute_loglike_for_sampling(self, tools: ModelingTools) -> float:
        """Return the sum of the components' sampling log-likelihoods.

        A component that converts a CCL error to -inf makes the sum -inf.

        :param tools: the ModelingTools passed to every component
        :returns: the sum of the log-likelihoods
        """
        return sum(
            likelihood.compute_loglike_for_sampling(tools)
            for likelihood in self.likelihoods
        )

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Return the sum of the components' log-likelihoods.

        :param tools: the ModelingTools passed to every component
        :returns: the sum of the log-likelihoods
        """
        return sum(likelihood.compute_loglike(tools) for likelihood in self.likelihoods)
