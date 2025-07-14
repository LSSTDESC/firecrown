import numpy as np
import pyccl
import pytest
import sacc
import firecrown.likelihood.likelihood as like


class LikelihoodThatThrowsIntegrationError(like.Likelihood):
    """A likelihood that always throws a pyccl integration error exception"."""

    def read(self, data: sacc.Sacc) -> None:
        pass

    def compute_loglike(self, _: like.ModelingTools) -> float:
        pyccl.pyutils.check(pyccl.lib.CCL_ERROR_INTEG)
        return -1.0


class LikelihoodThatThrowsUnhandledError(like.Likelihood):
    """A likelihood that always throws a pyccl integration error exception"."""

    def read(self, data: sacc.Sacc) -> None:
        pass

    def compute_loglike(self, _: like.ModelingTools) -> float:
        pyccl.pyutils.check(pyccl.lib.CCL_ERROR_FILE_READ)
        return -1.0


def test_integration_error_generates_0_likelihood():
    likelihood = LikelihoodThatThrowsIntegrationError()
    assert likelihood.compute_loglike_for_sampling(like.ModelingTools()) == -np.inf


def test_other_error_propagates():
    likelihood = LikelihoodThatThrowsUnhandledError()
    with pytest.raises(RuntimeError):
        likelihood.compute_loglike_for_sampling(like.ModelingTools())
