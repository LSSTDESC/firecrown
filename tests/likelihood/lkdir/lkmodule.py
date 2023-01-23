import sacc
import pyccl
from firecrown.parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from firecrown.likelihood.likelihood import Likelihood


class EmptyLikelihood(Likelihood):
    def __init__(self):
        super().__init__()
        self.placeholder = 1.0

    def read(self, sacc_data: sacc.Sacc):
        pass

    def _update(self, params: ParamsMap):
        pass

    def _reset(self) -> None:
        pass

    def _required_parameters(self):
        return RequiredParameters([])

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def compute_loglike(self, ccl_cosmo: pyccl.Cosmology) -> float:
        return -3.0 * self.placeholder


def empty_likelihood():
    return EmptyLikelihood()
