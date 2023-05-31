"""Unit tests for the numcosmo model connector."""

import pytest
import yaml
from firecrown.connector.numcosmo.model import (
    NumCosmoModel,
    ScalarParameter,
    VectorParameter,
    define_numcosmo_model,
)


@pytest.fixture(name="sparams")
def fixture_scalar_parameters():
    """Create a list of scalar parameters."""
    sp1 = ScalarParameter("s_1", "sp1", 0.0, 1.0, 0.1)
    sp2 = ScalarParameter("s_2", "sp2", -1.0, 2.0, 0.1)
    return [sp1, sp2]


@pytest.fixture(name="vparams")
def fixture_vector_parameters():
    """Create a list of vector parameters."""
    vp1 = VectorParameter(3, "v_1", "vp1", 0.0, 1.0, 0.1)
    vp2 = VectorParameter(4, "v_2", "vp2", 0.0, 1.0, 0.1)
    return [vp1, vp2]


@pytest.fixture(name="model")
def fixture_model(sparams, vparams):
    """Create a NumCosmoModel instance."""
    return NumCosmoModel("model1", "model1 description", sparams, vparams)


def test_parameterless_module_construction(model):
    """Test NumCosmoModel construction."""

    MyModel = define_numcosmo_model(model)

    m = MyModel()

    assert m is not None

    n_scalar_params = len(model.scalar_params)
    assert m.sparam_len() == n_scalar_params

    for i, sparam in enumerate(model.scalar_params):
        assert m.param_name(i) == sparam.name
        assert m.param_symbol(i) == sparam.symbol
        assert m.param_get(i) == sparam.default_value
        assert m.param_get_scale(i) == sparam.scale
        assert m.param_get_lower_bound(i) == sparam.lower_bound
        assert m.param_get_upper_bound(i) == sparam.upper_bound

    i = n_scalar_params
    for vparam in model.vector_params:
        for j in range(vparam.default_length):
            assert m.param_name(i) == f"{vparam.name}_{j}"
            assert m.param_symbol(i) == f"{{{vparam.symbol}}}_{j}"
            assert m.param_get(i) == vparam.default_value
            assert m.param_get_scale(i) == vparam.scale
            assert m.param_get_lower_bound(i) == vparam.lower_bound
            assert m.param_get_upper_bound(i) == vparam.upper_bound
            i += 1


def test_model_save_load(tmp_path, model):
    """Test saving and loading a NumCosmo model."""

    with open(
        tmp_path / r"numcosmo_firecrown_model.yml", "w", encoding="utf8"
    ) as modelfile:
        yaml.dump(model, modelfile, Dumper=yaml.Dumper)

    with open(
        tmp_path / r"numcosmo_firecrown_model.yml", "r", encoding="utf8"
    ) as modelfile:
        model_copy = yaml.load(modelfile, Loader=yaml.Loader)

    assert model_copy.name == model.name
    assert model_copy.description == model.description
    assert model_copy.scalar_params == model.scalar_params
    assert model_copy.vector_params == model.vector_params
