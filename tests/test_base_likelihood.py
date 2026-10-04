"""
Tests for the module firecrown.likelihood.likelihood
"""

import numpy as np
import pytest

import firecrown.likelihood._likelihood as lk


class MinimalLikelihood(lk.Likelihood):
    """A likelihood implementing every method of the base class trivially."""

    def compute_loglike(self, _):
        return -1.0

    def read(self, _):
        pass

    def make_realization_vector(self):
        return np.zeros(0)


def test_unimplemented_make_realization_vector():
    with pytest.raises(NotImplementedError):
        lk.Likelihood.make_realization_vector(MinimalLikelihood())
