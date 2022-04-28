import pytest
from firecrown.parameters import RequiredParameters


def test_get_params_names_does_not_allow_mutation():
    """The caller of RequiredParameters.get_params_names should not be able to modify the
    state of the object on which the call was made."""
    orig = RequiredParameters(["a", "b"])
    names = list(orig.get_params_names())
    assert names == ["a", "b"]
    names.append("c")
    assert list(orig.get_params_names()) == ["a", "b"]
