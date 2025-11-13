"""
Tests for the module firecrown.likelihood.likelihood
"""

import pytest
import firecrown.likelihood._likelihood as lk


class DefectiveLikelihood(lk.Likelihood):
    """This is a defective likelhood.

    It is lacking the required `make_realization_vector` method.
    """

    def compute_loglike(self, _):
        return -1.0

    def read(self, _):
        pass


def test_unimplemented_make_realization_vector():
    with pytest.raises(NotImplementedError):
        DefectiveLikelihood().make_realization_vector()
