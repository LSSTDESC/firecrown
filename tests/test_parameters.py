import pytest
import numpy as np
from firecrown.parameters import RequiredParameters, parameter_get_full_name, ParamsMap
from firecrown.parameters import (
    DerivedParameterScalar,
    DerivedParameterCollection,
)


def test_get_params_names_does_not_allow_mutation():
    """The caller of RequiredParameters.get_params_names should not be able to modify the
    state of the object on which the call was made."""
    orig = RequiredParameters(["a", "b"])
    names = set(orig.get_params_names())
    assert names == {"a", "b"}
    assert names == {"b", "a"}
    names.add("c")
    assert set(orig.get_params_names()) == {"a", "b"}


def test_params_map():
    my_params = ParamsMap({"a": 1})
    x = my_params.get_from_prefix_param(None, "a")
    assert x == 1
    with pytest.raises(KeyError):
        _ = my_params.get_from_prefix_param("no_such_prefix", "a")
    with pytest.raises(KeyError):
        _ = my_params.get_from_prefix_param(None, "no_such_name")


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
    full_name = parameter_get_full_name(None, "nomen_foo")
    assert full_name == "nomen_foo"


def test_derived_parameter_scalar():
    derived_param = DerivedParameterScalar("sec1", "name1", 3.14)

    assert isinstance(derived_param.get_val(), float)
    assert derived_param.get_val() == 3.14
    assert derived_param.get_full_name() == "sec1--name1"


def test_derived_parameter_wrong_type():
    """Try instantiating DerivedParameter objects with wrong types."""

    with pytest.raises(TypeError):
        derived_param = DerivedParameterScalar(
            "sec1", "name1", "not a float"
        )  # pylint: disable-msg=E0110,W0612
    with pytest.raises(TypeError):
        derived_param = DerivedParameterScalar(
            "sec1", "name1", [3.14]
        )  # pylint: disable-msg=E0110,W0612
    with pytest.raises(TypeError):
        derived_param = DerivedParameterScalar(
            "sec1", "name1", np.array([3.14])
        )  # pylint: disable-msg=E0110,W0612


def test_derived_parameters_collection():
    olist = [
        DerivedParameterScalar("sec1", "name1", 3.14),
        DerivedParameterScalar("sec2", "name2", 2.72),
    ]
    orig = DerivedParameterCollection(olist)
    clist = orig.get_derived_list()
    clist.append(DerivedParameterScalar("sec3", "name3", 0.58))
    assert orig.get_derived_list() == olist


def test_derived_parameters_collection_add():
    olist = [
        DerivedParameterScalar("sec1", "name1", 3.14),
        DerivedParameterScalar("sec2", "name2", 2.72),
        DerivedParameterScalar("sec2", "name3", 0.58),
    ]
    dpc1 = DerivedParameterCollection(olist)
    dpc2 = None

    dpc = dpc1 + dpc2

    for (section, name, val), derived_parameter in zip(dpc, olist):
        assert section == derived_parameter.section
        assert name == derived_parameter.name
        assert val == derived_parameter.get_val()


def test_derived_parameters_collection_add_iter():
    olist1 = [
        DerivedParameterScalar("sec1", "name1", 3.14),
        DerivedParameterScalar("sec2", "name2", 2.72),
        DerivedParameterScalar("sec2", "name3", 0.58),
    ]
    dpc1 = DerivedParameterCollection(olist1)

    olist2 = [
        DerivedParameterScalar("sec3", "name1", 3.14e1),
        DerivedParameterScalar("sec3", "name2", 2.72e1),
        DerivedParameterScalar("sec3", "name3", 0.58e1),
    ]
    dpc2 = DerivedParameterCollection(olist2)

    dpc = dpc1 + dpc2
    olist = olist1 + olist2

    for (section, name, val), derived_parameter in zip(dpc, olist):
        assert section == derived_parameter.section
        assert name == derived_parameter.name
        assert val == derived_parameter.get_val()
