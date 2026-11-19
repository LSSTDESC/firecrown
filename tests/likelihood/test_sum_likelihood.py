"""Tests for firecrown.likelihood.SumLikelihood and its use by the loader."""

import types

import pytest
import sacc

from firecrown import updatable
from firecrown.likelihood import (
    Likelihood,
    NamedParameters,
    SumLikelihood,
    load_likelihood_from_module_type,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap


class ConstantLikelihood(Likelihood):  # pylint: disable=abstract-method
    """Likelihood with a fixed log-likelihood that stores the SACC object read."""

    def __init__(
        self,
        loglike: float,
        raise_on_unused_parameter: bool = True,
    ) -> None:
        """Initialize with the log-likelihood value to return."""
        super().__init__(raise_on_unused_parameter=raise_on_unused_parameter)
        self.loglike = loglike
        self.sacc_data: None | sacc.Sacc = None

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Store the SACC object."""
        self.sacc_data = sacc_data

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Return the fixed log-likelihood."""
        return self.loglike


class SharedParameterLikelihood(ConstantLikelihood):  # pylint: disable=abstract-method
    """ConstantLikelihood with the sampler parameter `<prefix>_param0`."""

    def __init__(self, prefix: str) -> None:
        """Initialize with the parameter prefix."""
        super().__init__(0.0)
        self.parameter_prefix = prefix
        self.param0 = updatable.register_new_updatable_parameter(default_value=1.0)


def test_sum_likelihood_compute_loglike() -> None:
    """The log-likelihood is the sum of the components' log-likelihoods."""
    like = SumLikelihood([ConstantLikelihood(-3.0), ConstantLikelihood(-1.5)])
    tools = ModelingTools()

    assert like.compute_loglike(tools) == -4.5
    assert like.compute_loglike_for_sampling(tools) == -4.5


def test_sum_likelihood_accepts_iterable() -> None:
    """A generator of likelihoods is accepted."""
    like = SumLikelihood(ConstantLikelihood(-1.0) for _ in range(3))

    assert len(like.likelihoods) == 3
    assert like.compute_loglike(ModelingTools()) == -3.0


def test_sum_likelihood_empty() -> None:
    """An empty list of likelihoods raises ValueError."""
    with pytest.raises(ValueError, match="requires at least one likelihood"):
        SumLikelihood([])


def test_sum_likelihood_wrong_type() -> None:
    """A component that is not a Likelihood raises TypeError."""
    with pytest.raises(TypeError, match="must be a Firecrown `Likelihood`"):
        SumLikelihood([ConstantLikelihood(0.0), 1.0])  # type: ignore[list-item]


def test_sum_likelihood_read_forwards_to_components() -> None:
    """Every component reads the same SACC object."""
    components = [ConstantLikelihood(0.0), ConstantLikelihood(0.0)]
    like = SumLikelihood(components)
    sacc_data = sacc.Sacc()

    like.read(sacc_data)

    assert all(c.sacc_data is sacc_data for c in components)


def test_sum_likelihood_make_realization_vector() -> None:
    """The sum has no single data vector."""
    like = SumLikelihood([ConstantLikelihood(0.0)])

    with pytest.raises(NotImplementedError, match="no single data vector"):
        like.make_realization_vector()


@pytest.mark.parametrize(
    "flags, expected",
    [
        ((True, True), True),
        ((True, False), False),
        ((False, False), False),
    ],
)
def test_sum_likelihood_raise_on_unused_parameter(
    flags: tuple[bool, bool], expected: bool
) -> None:
    """Unused parameters raise only if every component requests it."""
    like = SumLikelihood([ConstantLikelihood(0.0, flag) for flag in flags])

    assert like.raise_on_unused_parameter is expected


def test_sum_likelihood_update_reset_shared_parameter() -> None:
    """A parameter shared by two components is required once and set in both."""
    components = [SharedParameterLikelihood("pre") for _ in range(2)]
    like = SumLikelihood(components)

    assert list(like.required_parameters().get_params_names()) == ["pre_param0"]

    like.update(ParamsMap({"pre_param0": 2.5}))
    assert like.is_updated()
    assert all(c.is_updated() for c in components)
    assert all(c.param0 == 2.5 for c in components)

    like.reset()
    assert not like.is_updated()
    assert not any(c.is_updated() for c in components)


def test_load_likelihood_from_list() -> None:
    """A factory returning a list of likelihoods yields a SumLikelihood."""
    module = types.ModuleType("sum_list_module")
    components: list[Likelihood] = [ConstantLikelihood(-3.0), ConstantLikelihood(-1.5)]
    module.build_likelihood = lambda _: components  # type: ignore[attr-defined]

    like, tools = load_likelihood_from_module_type(module, NamedParameters())

    assert isinstance(like, SumLikelihood)
    assert list(like.likelihoods) == components
    assert isinstance(tools, ModelingTools)
    assert like.compute_loglike(tools) == -4.5


def test_load_likelihood_from_list_with_tools() -> None:
    """A factory returning (list of likelihoods, tools) keeps the tools."""
    module = types.ModuleType("sum_list_tools_module")
    components: list[Likelihood] = [ConstantLikelihood(0.0), ConstantLikelihood(0.0)]
    tools = ModelingTools()
    module.build_likelihood = (  # type: ignore[attr-defined]
        lambda _: (components, tools)
    )

    like, loaded_tools = load_likelihood_from_module_type(module, NamedParameters())

    assert isinstance(like, SumLikelihood)
    assert list(like.likelihoods) == components
    assert loaded_tools is tools
