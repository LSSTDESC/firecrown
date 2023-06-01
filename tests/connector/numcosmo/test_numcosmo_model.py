"""Unit tests for the numcosmo model connector."""

import yaml
from firecrown.connector.numcosmo.model import (
    define_numcosmo_model,
)


def test_parameterless_module_construction(nc_model):
    """Test NumCosmoModel construction."""

    MyModel = define_numcosmo_model(nc_model)

    m = MyModel()

    assert m is not None

    n_scalar_params = len(nc_model.scalar_params)
    assert m.sparam_len() == n_scalar_params

    for i, sparam in enumerate(nc_model.scalar_params):
        assert m.param_name(i) == sparam.name
        assert m.param_symbol(i) == sparam.symbol
        assert m.param_get(i) == sparam.default_value
        assert m.param_get_scale(i) == sparam.scale
        assert m.param_get_lower_bound(i) == sparam.lower_bound
        assert m.param_get_upper_bound(i) == sparam.upper_bound

    i = n_scalar_params
    for vparam in nc_model.vector_params:
        for j in range(vparam.default_length):
            assert m.param_name(i) == f"{vparam.name}_{j}"
            assert m.param_symbol(i) == f"{{{vparam.symbol}}}_{j}"
            assert m.param_get(i) == vparam.default_value
            assert m.param_get_scale(i) == vparam.scale
            assert m.param_get_lower_bound(i) == vparam.lower_bound
            assert m.param_get_upper_bound(i) == vparam.upper_bound
            i += 1


def test_model_save_load(tmp_path, nc_model):
    """Test saving and loading a NumCosmo model."""

    with open(
        tmp_path / r"numcosmo_firecrown_model.yml", "w", encoding="utf8"
    ) as modelfile:
        yaml.dump(nc_model, modelfile, Dumper=yaml.Dumper)

    with open(
        tmp_path / r"numcosmo_firecrown_model.yml", "r", encoding="utf8"
    ) as modelfile:
        model_copy = yaml.load(modelfile, Loader=yaml.Loader)

    assert model_copy.name == nc_model.name
    assert model_copy.description == nc_model.description
    assert model_copy.scalar_params == nc_model.scalar_params
    assert model_copy.vector_params == nc_model.vector_params
