import numpy as np
import pyccl
import pytest
import sacc
import types
import firecrown.likelihood._likelihood as like
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood._likelihood import NamedParameters
from firecrown.parameters import ParamsMap


class LikelihoodThatThrowsIntegrationError(like.Likelihood):
    """A likelihood that always throws a pyccl integration error exception."""

    def read(self, sacc_data: sacc.Sacc) -> None:
        pass

    def compute_loglike(self, _: like.ModelingTools) -> float:
        pyccl.pyutils.check(pyccl.lib.CCL_ERROR_INTEG)
        return -1.0


class LikelihoodThatThrowsUnhandledError(like.Likelihood):
    """A likelihood that always throws a pyccl integration error exception."""

    def read(self, sacc_data: sacc.Sacc) -> None:
        pass

    def compute_loglike(self, _: like.ModelingTools) -> float:
        pyccl.pyutils.check(pyccl.lib.CCL_ERROR_FILE_READ)
        return -1.0


def test_integration_error_generates_0_likelihood():
    likelihood = LikelihoodThatThrowsIntegrationError()
    with pytest.warns(
        UserWarning,
        match="CCL error:\nError CCL_ERROR_INTEG: \nin likelihood, returning -inf",
    ):
        assert likelihood.compute_loglike_for_sampling(like.ModelingTools()) == -np.inf


def test_other_error_propagates():
    likelihood = LikelihoodThatThrowsUnhandledError()
    with pytest.raises(RuntimeError):
        likelihood.compute_loglike_for_sampling(like.ModelingTools())


def test_factory_returns_updated_likelihood():
    """Test warning when factory returns an already-updated likelihood."""

    class PreUpdatedLikelihood(like.Likelihood):
        """A likelihood that always returns 0 likelihood."""

        def read(self, sacc_data: sacc.Sacc) -> None:
            pass

        def compute_loglike(self, tools: ModelingTools) -> float:
            return 0.0

    def build_likelihood(_build_parameters: NamedParameters):
        """Factory that returns an updated likelihood."""
        likelihood = PreUpdatedLikelihood()
        # Pre-update the likelihood before returning it
        likelihood.update(ParamsMap())
        return likelihood

    # Create a proper module object with our factory
    module = types.ModuleType("test_module")
    module.build_likelihood = build_likelihood  # type: ignore
    module.__file__ = __file__  # Set the module's file attribute to this test file

    with pytest.warns(
        UserWarning, match=".*likelihood object that is already in an updated state.*"
    ):
        likelihood, _ = like.load_likelihood_from_module_type(module, NamedParameters())
        # Verify the likelihood was reset
        assert not likelihood.is_updated()


def test_factory_returns_updated_tools():
    """Test warning when factory returns already-updated tools."""

    class TestLikelihood(like.Likelihood):
        """A likelihood that always returns 0 likelihood."""

        def read(self, sacc_data: sacc.Sacc) -> None:
            pass

        def compute_loglike(self, tools: ModelingTools) -> float:
            return 0.0

    def build_likelihood(_build_parameters: NamedParameters):
        """Factory that returns an updated tools object."""
        likelihood = TestLikelihood()
        tools = ModelingTools()

        # Create params with required cosmological parameters
        params = ParamsMap()
        params.update(
            {
                "Omega_c": 0.27,  # dark matter density
                "Omega_b": 0.045,  # baryon density
                "h": 0.67,  # Hubble parameter
                "n_s": 0.96,  # spectral index
                "sigma8": 0.83,  # power spectrum amplitude
                "Omega_k": 0.0,  # curvature
                "w0": -1.0,  # dark energy equation of state
                "wa": 0.0,  # dark energy evolution
                "T_CMB": 2.7255,  # CMB temperature
                "Neff": 3.046,  # effective number of neutrinos
                "m_nu": 0.06,  # sum of neutrino masses
            }
        )

        # Pre-update the tools before returning them
        tools.update(params)
        return likelihood, tools

    # Create a proper module object with our factory
    module = types.ModuleType("test_module")
    module.build_likelihood = build_likelihood  # type: ignore
    module.__file__ = __file__  # Set the module's file attribute to this test file

    with pytest.warns(
        UserWarning, match=".*tools object that is already in an updated state.*"
    ):
        _, tools = like.load_likelihood_from_module_type(module, NamedParameters())
        # Verify the tools were reset
        assert not tools.is_updated()
