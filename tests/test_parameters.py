import pytest
from firecrown.parameters import RequiredParameters, parameter_get_full_name, ParamsMap


def test_get_params_names_does_not_allow_mutation():
    """The caller of RequiredParameters.get_params_names should not be able to modify the
    state of the object on which the call was made."""
    orig = RequiredParameters(["a", "b"])
    names = list(orig.get_params_names())
    assert names == ["a", "b"]
    names.append("c")
    assert list(orig.get_params_names()) == ["a", "b"]


def test_params_map():
    my_params = ParamsMap({"a": 1})
    x = my_params.get_from_prefix_param(None, "a")
    assert x == 1
    with pytest.raises(KeyError):
        _ = my_params.get_from_prefix_param("no_such_prefix", "a")
    with pytest.raises(KeyError):
        _ = my_params.get_from_prefix_param(None, "nosuchname")


def test_parameter_get_full_name_reject_empty_name():
    with pytest.raises(ValueError):
        _ = parameter_get_full_name(None, "")
    with pytest.raises(ValueError):
        _ = parameter_get_full_name("cow", "")


def test_parameter_get_full_name_with_prefix():
    full_name = parameter_get_full_name("my_prefix", "my_name")
    # TODO: do we really want to allow underscores in parameter names, when we
    # are using the underscore as our separator?
    assert full_name == "my_prefix_my_name"


def test_parameter_get_full_name_without_prefix():
    full_name = parameter_get_full_name(None, "nomen_mihi")
    assert full_name == "nomen_mihi"
