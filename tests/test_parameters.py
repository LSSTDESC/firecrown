"""
Tests for the module firecrown.parameters.
"""

import pytest
import numpy as np
from firecrown.parameters import RequiredParameters, parameter_get_full_name, ParamsMap
from firecrown.parameters import (
    DerivedParameter,
    DerivedParameterCollection,
    register_new_updatable_parameter,
    create,
    InternalParameter,
    SamplerParameter,
)


def test_create_with_no_arg():
    """Calling parameters.create() with no argument should return an
    SamplerParameter"""
    with pytest.deprecated_call():
        a_parameter = create()
        assert isinstance(a_parameter, SamplerParameter)


def test_create_with_float_arg():
    """Calling parameters.create() with a float argument should return a
    InternalParameter ."""
    with pytest.deprecated_call():
        a_parameter = create(1.5)
        assert isinstance(a_parameter, InternalParameter)
        assert a_parameter.value == 1.5


def test_register_new_updatable_parameter_with_no_arg():
    """Calling parameters.create() with no argument should return an
    SamplerParameter"""
    with pytest.deprecated_call():
        a_parameter = register_new_updatable_parameter()
    assert isinstance(a_parameter, SamplerParameter)


def test_register_new_updatable_parameter_with_float_arg():
    """Calling parameters.create() with a float argument should return a
    InternalParameter ."""
    a_parameter = register_new_updatable_parameter(1.5)
    assert isinstance(a_parameter, InternalParameter)
    assert a_parameter.get_value() == 1.5


def test_setting_internal_parameter():
    a_parameter = register_new_updatable_parameter(1.0)
    assert a_parameter.value == 1.0
    a_parameter.set_value(2.0)
    assert a_parameter.value == 2.0


def test_register_new_updatable_parameter_with_wrong_arg():
    """Calling parameters.create() with an org that is neither float nor None should
    raise a TypeError."""
    with pytest.raises(TypeError):
        _ = register_new_updatable_parameter("cow")  # type: ignore


def test_required_parameters_length():
    empty = RequiredParameters([])
    assert len(empty) == 0
    with pytest.deprecated_call():
        a = RequiredParameters([SamplerParameter(name="a")])
    assert len(a) == 1
    with pytest.deprecated_call():
        b = RequiredParameters([SamplerParameter(name="a"), SamplerParameter(name="b")])
    assert len(b) == 2


def test_required_parameters_equality_testing():
    with pytest.deprecated_call():
        a1 = RequiredParameters([SamplerParameter(name="a")])
        a2 = RequiredParameters([SamplerParameter(name="a")])
    assert a1 == a2
    assert a1 is not a2
    with pytest.deprecated_call():
        b = RequiredParameters([SamplerParameter(name="b")])
    assert a1 != b
    with pytest.raises(
        TypeError, match="Cannot compare a RequiredParameter to an object of type int"
    ):
        _ = a1 == 10


def test_get_params_names_does_not_allow_mutation():
    """The caller of RequiredParameters.get_params_names should not be able to modify
    the state of the object on which the call was made."""
    with pytest.deprecated_call():
        orig = RequiredParameters(
            [SamplerParameter(name="a"), SamplerParameter(name="b")]
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
    with pytest.deprecated_call():
        sp = SamplerParameter(name="name1")
    assert sp.name == "name1"
    assert sp.fullname == "name1"


def test_sampler_parameter_with_prefix():
    with pytest.deprecated_call():
        sp = SamplerParameter(name="name1", prefix="prefix1")
    assert sp.name == "name1"
    assert sp.fullname == "prefix1_name1"


def test_sampler_parameter_no_value():
    with pytest.deprecated_call():
        sp = SamplerParameter(name="name1")
    with pytest.raises(AssertionError):
        _ = sp.get_value()


def test_sampler_parameter_with_value():
    with pytest.deprecated_call():
        sp = SamplerParameter(name="name1")
    sp.set_value(3.14)
    assert sp.get_value() == 3.14


def test_sampler_parameter_with_value_and_prefix():
    with pytest.deprecated_call():
        sp = SamplerParameter(name="name1", prefix="prefix1")
    sp.set_value(3.14)
    assert sp.get_value() == 3.14


def test_sampler_parameter_with_value_and_prefix_and_fullname():
    with pytest.deprecated_call():
        sp = SamplerParameter(name="name1", prefix="prefix1")
    sp.set_value(3.14)
    assert sp.get_value() == 3.14
    assert sp.fullname == "prefix1_name1"


def test_sample_parameter_hash():
    with pytest.deprecated_call():
        sp1 = SamplerParameter(name="name1")
        sp2 = SamplerParameter(name="name1")
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


def test_sampler_parameter_default_value_no_default():
    with pytest.deprecated_call():
        sp = SamplerParameter(name="name1")
    assert sp.get_default_value() is None


def test_sampler_parameter_default_value_no_name():
    with pytest.deprecated_call():
        sp = SamplerParameter()
    with pytest.raises(ValueError, match="Parameter name is not set"):
        _ = sp.name


def test_required_parameters_get_names():
    with pytest.deprecated_call():
        rp = RequiredParameters(
            [SamplerParameter(name="name1"), SamplerParameter(name="name2")]
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


def test_required_parameters_get_default_values_no_default():
    with pytest.deprecated_call():
        rp = RequiredParameters(
            [
                SamplerParameter(name="name1"),
                SamplerParameter(name="name2", default_value=2.72),
                SamplerParameter(name="name3", default_value=2.3),
            ]
        )
    with pytest.raises(ValueError, match="Parameter name1 has no default value"):
        _ = rp.get_default_values()


def test_required_parameters_get_default_values_no_default3():
    with pytest.deprecated_call():
        rp = RequiredParameters(
            [
                SamplerParameter(name="name1", default_value=3.14),
                SamplerParameter(name="name2", default_value=2.72),
                SamplerParameter(name="name3"),
            ]
        )
    with pytest.raises(ValueError, match="Parameter name3 has no default value"):
        _ = rp.get_default_values()


def test_add_required_parameter():
    coll = DerivedParameterCollection([])
    coll.add_required_parameter(
        DerivedParameter(section="barnyard", name="cow", val=1.0)
    )
    assert len(coll) == 1

    with pytest.raises(
        ValueError,
        match="RequiredParameter named barnyard--cow is already present in the collection",
    ):
        coll.add_required_parameter(
            DerivedParameter(section="barnyard", name="cow", val=3.14)
        )
