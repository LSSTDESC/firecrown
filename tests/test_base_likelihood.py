"""
Tests for the module firecrown.likelihood.likelihood
"""

import numpy as np
import numpy.typing as npt
import pytest
import sacc

import firecrown.likelihood._likelihood as lk
from firecrown.modeling_tools import ModelingTools


class MinimalLikelihood(lk.Likelihood):
    """A likelihood implementing every method of the base class trivially."""

    def compute_loglike(self, _: ModelingTools) -> float:
        return -1.0

    def read(self, _: sacc.Sacc) -> None:
        pass

    def make_realization_vector(self) -> npt.NDArray[np.float64]:
        return np.zeros(0)


def test_unimplemented_make_realization_vector() -> None:
    with pytest.raises(NotImplementedError):
        lk.Likelihood.make_realization_vector(MinimalLikelihood())
