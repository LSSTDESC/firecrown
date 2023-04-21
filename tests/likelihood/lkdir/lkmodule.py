"""
Provides a trivial likelihood class and factory function for testing purposes.
"""
import sacc
from firecrown.parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from firecrown.likelihood.likelihood import Likelihood
from firecrown.modeling_tools import ModelingTools


class EmptyLikelihood(Likelihood):
    """Initialize the object with a placeholder value of 1."""

    def __init__(self) -> None:
        super().__init__()
        self.placeholder = 1.0

    def read(self, sacc_data: sacc.Sacc) -> None:
        """This class has nothing to read."""

    def _update(self, params: ParamsMap) -> None:
        """This class has no parameters to update."""

    def _reset(self) -> None:
        """This class has no state to reset."""

    def _required_parameters(self) -> RequiredParameters:
        """Return an empty RequiredParameters object."""
        return RequiredParameters([])

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return an empty DerivedParameterCollection."""
        return DerivedParameterCollection([])

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Return a constant value of the likelihood, determined by the value
        of self.placeholder."""
        return -3.0 * self.placeholder


def empty_likelihood() -> EmptyLikelihood:
    """Return an EmptyLikelihood object."""
    return EmptyLikelihood()
