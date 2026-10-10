"""
Tests for the module firecrown.likelihood.likelihood
"""

import pytest
import sacc

import firecrown.likelihood._likelihood as lk
from firecrown.modeling_tools import ModelingTools


class DefectiveLikelihood(lk.Likelihood):
    """This is a defective likelhood.

    It is lacking the required `make_realization_vector` method.
    """

    def compute_loglike(self, _: ModelingTools) -> float:
        return -1.0

    def read(self, _: sacc.Sacc) -> None:
        pass


def test_unimplemented_make_realization_vector() -> None:
    with pytest.raises(NotImplementedError):
        DefectiveLikelihood().make_realization_vector()
