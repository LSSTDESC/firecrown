"""
Provides a trivial likelihood class and factory function for testing purposes.
"""
import sacc
from firecrown.parameters import DerivedParameterCollection, DerivedParameterScalar
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown import parameters


class EmptyLikelihood(Likelihood):
    """A minimal likelihood for testing. This likelihood has no parameters."""

    def __init__(self) -> None:
        """Initialize the object with a placeholder value of 1."""
        super().__init__()
        self.placeholder = 1.0

    def read(self, sacc_data: sacc.Sacc) -> None:
        """This class has nothing to read."""

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Return a constant value of the likelihood, determined by the value
        of self.placeholder."""
        return -3.0 * self.placeholder

    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True
    ) -> sacc.Sacc:
        """This class does not implement make_realization."""
        raise NotImplementedError("This class does not implement make_realization.")


def empty_likelihood() -> EmptyLikelihood:
    """Return an EmptyLikelihood object."""
    return EmptyLikelihood()


class ParamaterizedLikelihood(Likelihood):
    """A minimal likelihood for testing. This likelihood requires a parameter
    named 'sacc_filename'."""

    def __init__(self, params: NamedParameters):
        """Initialize the ParameterizedLikelihood by reading the specificed
        sacc_filename value."""
        super().__init__()
        self.sacc_filename = params.get_string("sacc_filename")

    def read(self, sacc_data: sacc.Sacc) -> None:
        """This class has nothing to read."""

    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True
    ) -> sacc.Sacc:
        """This class does not implement make_realization."""
        raise NotImplementedError("This class does not implement make_realization.")

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Return a constant value of the likelihood."""
        return -1.5


class SamplerParameterLikelihood(Likelihood):
    """A minimal likelihood for testing. This likelihood requires a parameter
    named 'sacc_filename'."""

    def __init__(self, params: NamedParameters):
        """Initialize the SamplerParameterLikelihood by reading the specificed
        parameter_prefix value and creates a sampler parameter called "sampler_param0".
        """
        super().__init__(parameter_prefix=params.get_string("parameter_prefix"))
        self.sampler_param0 = parameters.register_new_updatable_parameter()

    def read(self, sacc_data: sacc.Sacc) -> None:
        """This class has nothing to read."""

    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True
    ) -> sacc.Sacc:
        """This class does not implement make_realization."""
        raise NotImplementedError("This class does not implement make_realization.")

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Return a constant value of the likelihood."""
        return -2.1


class DerivedParameterLikelihood(Likelihood):
    """A minimal likelihood for testing. This likelihood requires a parameter
    named 'sacc_filename'."""

    def __init__(self):
        """Initialize the DerivedParameterLikelihood where _get_derived_parameters
        creates a derived parameter called "derived_param0".
        """
        super().__init__()
        self.placeholder = 1.0

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection(
            [DerivedParameterScalar("derived_section", "derived_param0", 1.0)]
        )

    def read(self, sacc_data: sacc.Sacc) -> None:
        """This class has nothing to read."""

    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True
    ) -> sacc.Sacc:
        """This class does not implement make_realization."""
        raise NotImplementedError("This class does not implement make_realization.")

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Return a constant value of the likelihood."""
        return -3.14


def parameterized_likelihood(params: NamedParameters):
    """Return a ParameterizedLikelihood object."""
    return ParamaterizedLikelihood(params)


def sampler_parameter_likelihood(params: NamedParameters):
    """Return a SamplerParameterLikelihood object."""
    return SamplerParameterLikelihood(params)


def derived_parameter_likelihood():
    """Return a DerivedParameterLikelihood object."""
    return DerivedParameterLikelihood()
