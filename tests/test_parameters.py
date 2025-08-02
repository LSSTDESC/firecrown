"""
Tests for the module firecrown.parameters.
"""

import re
import pytest
import numpy as np
from firecrown.parameters import (
    RequiredParameters,
    parameter_get_full_name,
    ParamsMap,
    handle_unused_params,
)
from firecrown.parameters import (
    DerivedParameter,
    DerivedParameterCollection,
    register_new_updatable_parameter,
    InternalParameter,
    SamplerParameter,
)


def test_register_new_updatable_parameter_with_no_arg():
    """Calling parameters.create() with no argument should return an
    SamplerParameter"""
    a_parameter = register_new_updatable_parameter(default_value=1.0)
    assert isinstance(a_parameter, SamplerParameter)


def test_register_new_updatable_parameter_with_float_arg():
    """Calling parameters.create() with a float argument should return a
    InternalParameter ."""
    a_parameter = register_new_updatable_parameter(value=1.5, default_value=1.0)
    assert isinstance(a_parameter, InternalParameter)
    assert a_parameter.get_value() == 1.5


def test_register_new_updatable_parameter_with_wrong_arg():
    """Calling parameters.create() with an org that is neither float nor None should
    raise a TypeError."""
    with pytest.raises(TypeError):
        _ = register_new_updatable_parameter(
            value="cow", default_value="moo"  # type: ignore
        )


def test_required_parameters_length():
    empty = RequiredParameters([])
    assert len(empty) == 0
    a = RequiredParameters([SamplerParameter(name="a", default_value=3.14)])
    assert len(a) == 1
    b = RequiredParameters(
        [
            SamplerParameter(name="a", default_value=3.14),
            SamplerParameter(name="b", default_value=3),
        ]
    )
    assert len(b) == 2


def test_required_parameters_equality_testing():
    a1 = RequiredParameters([SamplerParameter(name="a", default_value=2)])
    a2 = RequiredParameters([SamplerParameter(name="a", default_value=2)])
    assert a1 == a2
    assert a1 is not a2
    b = RequiredParameters([SamplerParameter(name="b", default_value=3)])
    assert a1 != b
    with pytest.raises(
        TypeError, match="Cannot compare a RequiredParameter to an object of type int"
    ):
        _ = a1 == 10


def test_get_params_names_does_not_allow_mutation():
    """The caller of RequiredParameters.get_params_names should not be able to modify
    the state of the object on which the call was made."""
    orig = RequiredParameters(
        [
            SamplerParameter(name="a", default_value=1),
            SamplerParameter(name="b", default_value=2),
        ]
    )
    names = set(orig.get_params_names())
    assert names == {"a", "b"}
    assert names == {"b", "a"}
    names.add("c")
    assert set(orig.get_params_names()) == {"a", "b"}


def test_params_map():
    my_params = ParamsMap({"a": 1.0})
    x = my_params.get_from_prefix_param(None, "a")
    assert x == 1
    with pytest.raises(KeyError):
        _ = my_params.get_from_prefix_param("no_such_prefix", "a")
    with pytest.raises(KeyError):
        _ = my_params.get_from_prefix_param(None, "no_such_name")


def test_params_map_wrong_type():
    with pytest.raises(
        TypeError, match="Value for parameter a is not a float or a list of floats.*"
    ):
        _ = ParamsMap({"a": "not a float or a list of floats"})


def test_params_map_wrong_type_list():
    with pytest.raises(
        TypeError, match="Value for parameter a is not a float or a list of floats.*"
    ):
        _ = ParamsMap({"a": ["not a float or a list of floats"]})


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
    derived_param = DerivedParameter("sec1", "name1", 3.14)

    assert isinstance(derived_param.get_val(), float)
    assert derived_param.get_val() == 3.14
    assert derived_param.get_full_name() == "sec1--name1"


def test_derived_parameter_wrong_type():
    """Try instantiating DerivedParameter objects with wrong types."""

    with pytest.raises(TypeError):
        _ = DerivedParameter(  # pylint: disable-msg=E0110,W0612
            "sec1", "name1", "not a float"  # type: ignore
        )
    with pytest.raises(TypeError):
        _ = DerivedParameter(  # pylint: disable-msg=E0110,W0612
            "sec1", "name1", [3.14]  # type: ignore
        )
    with pytest.raises(TypeError):
        _ = DerivedParameter(  # pylint: disable-msg=E0110,W0612
            "sec1", "name1", np.array([3.14])  # type: ignore
        )


def test_derived_parameters_collection():
    olist = [
        DerivedParameter("sec1", "name1", 3.14),
        DerivedParameter("sec2", "name2", 2.72),
    ]
    orig = DerivedParameterCollection(olist)
    clist = orig.get_derived_list()
    clist.append(DerivedParameter("sec3", "name3", 0.58))
    assert orig.get_derived_list() == olist


def test_derived_parameters_collection_rejects_bad_list():
    badlist = [1, 3, 5]
    with pytest.raises(TypeError):
        # We have to tell mypy to ignore the type error on the
        # next line, because it is the very type error we are
        # testing.
        _ = DerivedParameterCollection(badlist)  # type: ignore


def test_derived_parameters_collection_add():
    olist = [
        DerivedParameter("sec1", "name1", 3.14),
        DerivedParameter("sec2", "name2", 2.72),
        DerivedParameter("sec2", "name3", 0.58),
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
        DerivedParameter("sec1", "name1", 3.14),
        DerivedParameter("sec2", "name2", 2.72),
        DerivedParameter("sec2", "name3", 0.58),
    ]
    dpc1 = DerivedParameterCollection(olist1)

    olist2 = [
        DerivedParameter("sec3", "name1", 3.14e1),
        DerivedParameter("sec3", "name2", 2.72e1),
        DerivedParameter("sec3", "name3", 0.58e1),
    ]
    dpc2 = DerivedParameterCollection(olist2)

    dpc = dpc1 + dpc2
    olist = olist1 + olist2

    for (section, name, val), derived_parameter in zip(dpc, olist):
        assert section == derived_parameter.section
        assert name == derived_parameter.name
        assert val == derived_parameter.get_val()


def test_derived_parameter_eq():
    dv1 = DerivedParameter("sec1", "name1", 3.14)
    dv2 = DerivedParameter("sec1", "name1", 3.14)

    assert dv1 == dv2


def test_derived_parameter_eq_invalid():
    dv1 = DerivedParameter("sec1", "name1", 3.14)

    with pytest.raises(
        NotImplementedError,
        match="DerivedParameter comparison is only "
        "implemented for DerivedParameter objects",
    ):
        _ = dv1 == 1.0


def test_derived_parameters_collection_eq():
    olist1 = [
        DerivedParameter("sec1", "name1", 3.14),
        DerivedParameter("sec2", "name2", 2.72),
        DerivedParameter("sec2", "name3", 0.58),
    ]
    dpc1 = DerivedParameterCollection(olist1)

    olist2 = [
        DerivedParameter("sec1", "name1", 3.14),
        DerivedParameter("sec2", "name2", 2.72),
        DerivedParameter("sec2", "name3", 0.58),
    ]
    dpc2 = DerivedParameterCollection(olist2)

    assert dpc1 == dpc2


def test_derived_parameters_collection_eq_invalid():
    olist1 = [
        DerivedParameter("sec1", "name1", 3.14),
        DerivedParameter("sec2", "name2", 2.72),
        DerivedParameter("sec2", "name3", 0.58),
    ]
    dpc1 = DerivedParameterCollection(olist1)

    with pytest.raises(
        NotImplementedError,
        match="DerivedParameterCollection comparison is only "
        "implemented for DerivedParameterCollection objects",
    ):
        _ = dpc1 == 1.0


def test_sampler_parameter_no_prefix():
    sp = SamplerParameter(name="name1", default_value=1)
    assert sp.name == "name1"
    assert sp.fullname == "name1"


def test_sampler_parameter_with_prefix():
    sp = SamplerParameter(name="name1", prefix="prefix1", default_value=1)
    assert sp.name == "name1"
    assert sp.fullname == "prefix1_name1"


def test_sampler_parameter_with_value():
    sp = SamplerParameter(name="name1", default_value=2)
    assert sp is not None


def test_sampler_parameter_with_value_and_prefix():
    sp = SamplerParameter(name="name1", prefix="prefix1", default_value=2)
    assert sp is not None


def test_sampler_parameter_with_value_and_prefix_and_fullname():
    sp = SamplerParameter(name="name1", prefix="prefix1", default_value=2)
    assert sp.fullname == "prefix1_name1"


def test_sample_parameter_hash():
    sp1 = SamplerParameter(name="name1", default_value=3.14)
    sp2 = SamplerParameter(name="name1", default_value=3.14)
    assert hash(sp1) == hash(sp2)


def test_sampler_parameter_eq():
    sp1 = SamplerParameter(name="name1", default_value=3.14)
    sp2 = SamplerParameter(name="name1", default_value=3.14)

    assert sp1 == sp2


def test_sampler_parameter_eq_invalid():
    sp1 = SamplerParameter(name="name1", default_value=3.14)
    with pytest.raises(
        NotImplementedError,
        match=(
            "SamplerParameter comparison is only implemented for "
            "SamplerParameter objects"
        ),
    ):
        _ = sp1 == 1.0


def test_sampler_parameter_ne():
    sp1 = SamplerParameter(name="name1", default_value=3.14)
    sp2 = SamplerParameter(name="name2", default_value=3.14)

    assert sp1 != sp2


def test_sampler_parameter_default_value():
    sp = SamplerParameter(name="name1", default_value=3.14)
    assert sp.get_default_value() == 3.14


def test_sampler_parameter_default_value_no_name():
    sp = SamplerParameter(default_value=2.72)
    with pytest.raises(ValueError, match="Parameter name is not set"):
        _ = sp.name


def test_required_parameters_get_names():
    rp = RequiredParameters(
        [
            SamplerParameter(name="name1", default_value=3.14),
            SamplerParameter(name="name2", default_value=2.72),
        ]
    )
    assert set(rp.get_params_names()) == {"name1", "name2"}


def test_required_parameters_get_default_values():
    rp = RequiredParameters(
        [
            SamplerParameter(name="name1", default_value=3.14),
            SamplerParameter(name="name2", default_value=2.72),
            SamplerParameter(name="name3", default_value=2.3),
        ]
    )
    assert rp.get_default_values() == {"name1": 3.14, "name2": 2.72, "name3": 2.3}


def test_add_required_parameter():
    coll = DerivedParameterCollection([])
    coll.add_required_parameter(
        DerivedParameter(section="barnyard", name="cow", val=1.0)
    )
    assert len(coll) == 1

    with pytest.raises(
        ValueError,
        match="RequiredParameter named barnyard--cow is already present",
    ):
        coll.add_required_parameter(
            DerivedParameter(section="barnyard", name="cow", val=3.14)
        )


def test_setting_internal_parameter():
    a_parameter = register_new_updatable_parameter(value=1.0, default_value=2.0)
    assert a_parameter.value == 1.0
    a_parameter.set_value(2.0)
    assert a_parameter.value == 2.0


def test_used_and_unused_params():
    """Test that get_unused_keys works as expected."""
    params = ParamsMap({"a": 1.0, "b": 2.0, "c": 3.0})

    unused = params.get_unused_keys()

    assert unused == {"a", "b", "c"}

    params.get_from_full_name("a")
    unused = params.get_unused_keys()

    assert unused == {"b", "c"}

    params.get_from_full_name("b")
    unused = params.get_unused_keys()

    assert unused == {"c"}

    params.get_from_full_name("c")
    unused = params.get_unused_keys()
    assert unused == set()

    params.get_from_full_name("a")  # Accessing again should not change unused
    unused = params.get_unused_keys()
    assert unused == set()


def test_handle_unused_params():
    """Test that handle_unused_params works as expected."""
    params = ParamsMap({"a": 1.0, "b": 2.0, "c": 3.0})

    # All parameters are unused, should raise a warning
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Unused keys in parameters: ['a', 'b', 'c']. "
            "This may indicate a problem with the parameter mapping."
        ),
    ):
        handle_unused_params(params=params, raise_on_unused=False)

    # All parameters are unused, should raise an error
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unused keys in parameters: ['a', 'b', 'c']. "
            "This may indicate a problem with the parameter mapping."
        ),
    ):
        handle_unused_params(params=params, raise_on_unused=True)

    # Use some parameters
    params.get_from_full_name("a")
    params.get_from_full_name("b")

    # Now 'c' is unused, should raise a warning
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Unused keys in parameters: ['c']. "
            "This may indicate a problem with the parameter mapping."
        ),
    ):
        handle_unused_params(params=params, raise_on_unused=False)

    # Now 'c' is unused, should raise an error
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unused keys in parameters: ['c']. "
            "This may indicate a problem with the parameter mapping."
        ),
    ):
        handle_unused_params(params=params, raise_on_unused=True)
