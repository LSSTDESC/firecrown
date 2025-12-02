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
    DerivedParameter,
    DerivedParameterCollection,
    register_new_updatable_parameter,
    InternalParameter,
    SamplerParameter,
)
from firecrown.updatable import (
    Updatable,
    UpdatableUsageRecord,
    get_default_params,
    get_default_params_map,
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


def test_register_new_updatable_parameter_with_shared_false():
    """Calling parameters.create() with shared=False should return a SamplerParameter
    with shared=False."""
    a_parameter = register_new_updatable_parameter(default_value=1.0, shared=False)
    assert isinstance(a_parameter, SamplerParameter)
    assert a_parameter.shared is False


def test_register_new_updatable_parameter_with_shared_true():
    """Calling parameters.create() with shared=True (default) should return a
    SamplerParameter with shared=True."""
    a_parameter = register_new_updatable_parameter(default_value=1.0, shared=True)
    assert isinstance(a_parameter, SamplerParameter)
    assert a_parameter.shared is True

    # Also test the default behavior
    a_parameter_default = register_new_updatable_parameter(default_value=1.0)
    assert a_parameter_default.shared is True


def test_sampler_parameter_shared_ignores_prefix():
    """When shared=False, the SamplerParameter should ignore the prefix in
    set_fullname."""
    sp = SamplerParameter(default_value=1.0, shared=False)
    sp.set_fullname("my_prefix", "my_name")

    # With shared=False, prefix should be ignored
    assert sp.prefix is None
    assert sp.name == "my_name"
    assert sp.fullname == "my_name"


def test_sampler_parameter_shared_uses_prefix():
    """When shared=True (default), the SamplerParameter should use the prefix
    in set_fullname."""
    sp = SamplerParameter(default_value=1.0, shared=True)
    sp.set_fullname("my_prefix", "my_name")

    # With shared=True, prefix should be used
    assert sp.prefix == "my_prefix"
    assert sp.name == "my_name"
    assert sp.fullname == "my_prefix_my_name"


def test_sampler_parameter_equality_with_shared():
    """Test that SamplerParameter equality considers the shared attribute."""
    sp1 = SamplerParameter(default_value=1.0, name="a", shared=True)
    sp2 = SamplerParameter(default_value=1.0, name="a", shared=True)
    sp3 = SamplerParameter(default_value=1.0, name="a", shared=False)

    assert sp1 == sp2
    assert sp1 != sp3


def test_shared_parameter_across_updatable_instances():
    """Test that shared=False parameters are consistent across multiple
    instances of the same Updatable class.

    When shared=False, the parameter should not receive a prefix, making it
    the same across all instances. Both instances should receive the same
    value from the ParamsMap when updated.
    """

    class UpdatableWithSharedParam(Updatable):
        """An Updatable with a shared (non-prefixed) parameter."""

        def __init__(self, prefix: str | None = None):
            super().__init__(prefix)
            # This parameter is NOT shared (shared=False), so it won't get a prefix
            self.global_param = register_new_updatable_parameter(
                default_value=1.0, shared=False
            )
            # This parameter IS shared (default), so it will get a prefix
            self.local_param = register_new_updatable_parameter(default_value=2.0)

    # Create two instances with different prefixes
    instance1 = UpdatableWithSharedParam("inst1")
    instance2 = UpdatableWithSharedParam("inst2")

    # Verify that the non-shared parameter has no prefix in both instances
    # pylint: disable=protected-access
    assert instance1._sampler_parameters[0].fullname == "global_param"
    assert instance2._sampler_parameters[0].fullname == "global_param"

    # Verify that the shared parameter has different prefixes
    assert instance1._sampler_parameters[1].fullname == "inst1_local_param"
    assert instance2._sampler_parameters[1].fullname == "inst2_local_param"
    # pylint: enable=protected-access

    # Create a ParamsMap with a single value for global_param and different
    # values for each instance's local_param
    params = ParamsMap(
        {
            "global_param": 42.0,
            "inst1_local_param": 10.0,
            "inst2_local_param": 20.0,
        }
    )

    # Update both instances
    instance1.update(params)
    instance2.update(params)

    # Both instances should have the same value for global_param
    assert instance1.global_param == 42.0
    assert instance2.global_param == 42.0
    assert instance1.global_param == instance2.global_param

    # But different values for local_param
    assert instance1.local_param == 10.0
    assert instance2.local_param == 20.0


def test_get_default_params_with_shared_parameters():
    """Test that get_default_params and get_default_params_map work correctly
    when two instances of the same Updatable class share a parameter.

    When shared=False, both instances will have the same parameter name without
    a prefix. The get_default_params should return a single entry for the shared
    parameter and separate entries for the prefixed parameters.
    """

    class UpdatableWithSharedParam(Updatable):
        """An Updatable with a shared (non-prefixed) parameter."""

        def __init__(self, prefix: str | None = None):
            super().__init__(prefix)
            # This parameter is NOT shared (shared=False), so it won't get a prefix
            self.global_param = register_new_updatable_parameter(
                default_value=1.0, shared=False
            )
            # This parameter IS shared (default), so it will get a prefix
            self.local_param = register_new_updatable_parameter(default_value=2.0)

    # Create two instances with different prefixes
    instance1 = UpdatableWithSharedParam("inst1")
    instance2 = UpdatableWithSharedParam("inst2")

    # Get default parameters from both instances
    default_params = get_default_params(instance1, instance2)

    # Should have 3 entries:
    # - "global_param" (shared, appears once)
    # - "inst1_local_param"
    # - "inst2_local_param"
    assert len(default_params) == 3
    assert "global_param" in default_params
    assert "inst1_local_param" in default_params
    assert "inst2_local_param" in default_params

    # Check default values
    assert default_params["global_param"] == 1.0
    assert default_params["inst1_local_param"] == 2.0
    assert default_params["inst2_local_param"] == 2.0

    # Test get_default_params_map as well
    params_map = get_default_params_map(instance1, instance2)
    assert params_map.get_from_full_name("global_param") == 1.0
    assert params_map.get_from_full_name("inst1_local_param") == 2.0
    assert params_map.get_from_full_name("inst2_local_param") == 2.0

    # Verify we can update both instances with the default params map
    instance1.update(params_map)
    instance2.update(params_map)

    # Both should have the same value for global_param
    assert instance1.global_param == 1.0
    assert instance2.global_param == 1.0


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


def test_params_map_getitem():
    a = ParamsMap({"a": 1.0})
    assert a.get_unused_keys() == {"a"}
    assert a["a"] == 1
    assert a.used_keys == {"a"}
    assert a.get_unused_keys() == set()
    with pytest.raises(KeyError):
        _ = a["b"]


def test_get_uses_params():
    a = ParamsMap({"a": 1.0})
    assert a.used_keys == set()
    v = a.get("no_such_key", -1)
    assert v == -1
    assert a.used_keys == set()
    v2 = a.get("a", 2.0)
    assert v2 == 1.0
    assert a.used_keys == {"a"}


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
        match=re.escape("Unused keys in parameters: ['a', 'b', 'c']."),
    ):
        handle_unused_params(params=params, updated_records=[], raise_on_unused=False)

    # All parameters are unused, should raise an error
    with pytest.raises(
        ValueError,
        match=re.escape("Unused keys in parameters: ['a', 'b', 'c']."),
    ):
        handle_unused_params(params=params, updated_records=[], raise_on_unused=True)

    # Use some parameters
    params.get_from_full_name("a")
    params.get_from_full_name("b")

    # Now 'c' is unused, should raise a warning
    with pytest.warns(
        UserWarning,
        match=re.escape("Unused keys in parameters: ['c']."),
    ):
        handle_unused_params(params=params, updated_records=[], raise_on_unused=False)

    # Now 'c' is unused, should raise an error
    with pytest.raises(
        ValueError,
        match=re.escape("Unused keys in parameters: ['c']."),
    ):
        handle_unused_params(params=params, updated_records=[], raise_on_unused=True)


def test_handle_unused_params_includes_updated_records_in_message():
    params = ParamsMap({"x": 1.0})

    # Create a simple UpdatableUsageRecord structure
    child = UpdatableUsageRecord(
        cls="Child",
        prefix="c",
        obj_id=2,
        sampler_params=["a"],
        internal_params=[],
        child_records=[],
    )
    parent = UpdatableUsageRecord(
        cls="Parent",
        prefix="p",
        obj_id=1,
        sampler_params=[],
        internal_params=[],
        child_records=[child],
    )

    updated_records = [parent]

    # Expect a warning that contains the unused key and the log lines from the record
    with pytest.warns(UserWarning) as record:
        handle_unused_params(
            params=params, updated_records=updated_records, raise_on_unused=False
        )

    msg = str(record[0].message)
    assert "Unused keys in parameters: ['x']." in msg
    # ensure the appended log lines are present
    assert "Parent(p) => Child(c):" in msg or "Parent(p):" in msg

    # Now test raise_on_unused=True includes the same information in the exception
    with pytest.raises(ValueError) as exc:
        handle_unused_params(
            params=params, updated_records=updated_records, raise_on_unused=True
        )

    err_msg = str(exc.value)
    assert "Unused keys in parameters: ['x']." in err_msg
    assert "Parent(p) => Child(c):" in err_msg or "Parent(p):" in err_msg


def test_params_map_union():
    p1 = ParamsMap({"a": 1.0})
    p2 = ParamsMap({"b": 2.0})
    p3 = p1.union(p2)
    assert p3.used_keys == set()
    assert p3.get_from_prefix_param(None, "a") == 1.0
    assert p3.used_keys == {"a"}
    assert p3.get_from_prefix_param(None, "b") == 2.0
    assert p3.used_keys == {"a", "b"}

    p4 = ParamsMap({"d": 3.0})
    p5 = p3.union(p4)
    assert p5.used_keys == {"a", "b"}

    assert p4["d"] == 3.0
    assert p4.used_keys == {"d"}
    p6 = p3.union(p4)
    assert p6.used_keys == {"a", "b", "d"}

    with pytest.raises(
        ValueError, match="Key a has different values in self and other."
    ):
        p1 = ParamsMap({"a": 1.0})
        p2 = ParamsMap({"a": 2.0})
        p1.union(p2)


def test_params_get():
    params = ParamsMap({"a": 1.0})
    assert params.get("a") == 1.0
    assert params.get("b", 2.0) == 2.0
    with pytest.raises(KeyError):
        params.get("b")


def test_params_map_items():
    d = {"a": 1.0, "b": 2.0}
    params = ParamsMap(d)
    assert params.items() == d.items()


def test_params_map_lower_case_lookup():
    """Ensure get_from_full_name respects the lower-case fallback when enabled."""
    p = ParamsMap({"my_key": 4.2})
    # lookup with different case should fail until we enable lower-case handling
    with pytest.raises(KeyError):
        _ = p.get_from_full_name("MY_KEY")

    p.use_lower_case_keys(True)
    # now the upper-case query should find the lower-case stored key
    val = p.get_from_full_name("MY_KEY")
    assert val == 4.2


def test_params_map_list_with_ints_raises_typeerror():
    """A list value containing ints (not floats) should trigger the list
    element type check and raise TypeError."""
    with pytest.raises(TypeError, match="Value for parameter a is not a float"):
        _ = ParamsMap({"a": [1, 2.0]})


def test_params_map_setitem():
    """Test ParamsMap.__setitem__() method for setting values."""
    params = ParamsMap({"a": 1.0})
    params["b"] = 2.0
    assert params["b"] == 2.0
    params["a"] = 3.0
    assert params["a"] == 3.0


def test_params_map_contains():
    """Test ParamsMap.__contains__() method with in operator."""
    params = ParamsMap({"a": 1.0, "b": 2.0})
    assert "a" in params
    assert "b" in params
    assert "c" not in params
    assert "nonexistent" not in params


def test_params_map_copy():
    """Test ParamsMap.copy() creates an independent copy."""
    original = ParamsMap({"a": 1.0, "b": 2.0})
    _ = original["a"]  # Mark 'a' as used
    assert original.used_keys == {"a"}

    # Create copy
    copied = original.copy()

    # Verify copy has same data and same used_keys state
    assert copied.params == {"a": 1.0, "b": 2.0}
    assert copied.used_keys == {"a"}
    assert copied.lower_case == original.lower_case

    # Verify independence: modifying copy doesn't affect original
    copied["c"] = 3.0
    assert "c" in copied
    assert "c" not in original

    copied.used_keys.add("b")
    assert "b" not in original.used_keys

    copied.lower_case = True
    assert original.lower_case is False


def test_params_map_update_duplicate_key():
    """Test ParamsMap.update() raises error for duplicate keys."""
    params = ParamsMap({"a": 1.0})

    # Update with new key should work
    params.update({"b": 2.0})
    assert params["b"] == 2.0

    # Update with duplicate key should raise ValueError
    with pytest.raises(ValueError, match="Key a is already present in the ParamsMap"):
        params.update({"a": 3.0})


def test_params_map_keys():
    """Test ParamsMap.keys() returns correct set of keys."""
    params = ParamsMap({"a": 1.0, "b": 2.0, "c": 3.0})
    keys = params.keys()

    assert isinstance(keys, set)
    assert keys == {"a", "b", "c"}

    # Verify returned set is independent (modifying doesn't affect ParamsMap)
    keys.add("d")
    assert params.keys() == {"a", "b", "c"}


def test_required_parameters_subtraction():
    """Test RequiredParameters.__sub__() subtraction operator."""
    param_a = SamplerParameter(name="a", default_value=1.0)
    param_b = SamplerParameter(name="b", default_value=2.0)
    param_c = SamplerParameter(name="c", default_value=3.0)
    param_d = SamplerParameter(name="d", default_value=4.0)

    params1 = RequiredParameters([param_a, param_b, param_c])
    params2 = RequiredParameters([param_b, param_d])

    # Subtract params2 from params1
    result = params1 - params2

    # Result should have 'a' and 'c' (not 'b' which was in params2)
    result_names = set(result.get_params_names())
    assert result_names == {"a", "c"}

    # Verify original objects unchanged
    assert set(params1.get_params_names()) == {"a", "b", "c"}
    assert set(params2.get_params_names()) == {"b", "d"}


def test_required_parameters_addition():
    """Test RequiredParameters.__add__() addition operator."""
    param_a = SamplerParameter(name="a", default_value=1.0)
    param_b = SamplerParameter(name="b", default_value=2.0)
    param_c = SamplerParameter(name="c", default_value=3.0)

    params1 = RequiredParameters([param_a, param_b])
    params2 = RequiredParameters([param_c])

    # Add params2 to params1
    result = params1 + params2

    # Result should have all three parameters
    result_names = set(result.get_params_names())
    assert result_names == {"a", "b", "c"}

    # Verify original objects unchanged
    assert set(params1.get_params_names()) == {"a", "b"}
    assert set(params2.get_params_names()) == {"c"}


def test_sampler_parameter_set_fullname():
    """Test SamplerParameter.set_fullname() method."""
    sp = SamplerParameter(default_value=1.0)

    # Initially no name/prefix set, accessing name should raise
    with pytest.raises(ValueError, match="Parameter name is not set"):
        _ = sp.name

    # Set fullname with prefix
    sp.set_fullname("my_prefix", "my_name")
    assert sp.name == "my_name"
    assert sp.prefix == "my_prefix"
    assert sp.fullname == "my_prefix_my_name"

    # Set fullname with None prefix
    sp.set_fullname(None, "another_name")
    assert sp.name == "another_name"
    assert sp.prefix is None
    assert sp.fullname == "another_name"


def test_params_map_union_no_common_keys():
    """Test ParamsMap.union() when there are no common keys.

    This covers the branch where the for loop over common keys is not entered.
    """
    p1 = ParamsMap({"a": 1.0, "b": 2.0})
    p2 = ParamsMap({"c": 3.0, "d": 4.0})

    # Union with no overlapping keys
    p3 = p1.union(p2)

    assert p3.get_from_full_name("a") == 1.0
    assert p3.get_from_full_name("b") == 2.0
    assert p3.get_from_full_name("c") == 3.0
    assert p3.get_from_full_name("d") == 4.0


def test_params_map_union_common_keys_same_values():
    """Test ParamsMap.union() when common keys have the same values.

    This covers the branch where common keys exist but don't trigger
    the ValueError because they have matching values.
    """
    p1 = ParamsMap({"a": 1.0, "b": 2.0})
    p2 = ParamsMap({"b": 2.0, "c": 3.0})

    # Union with overlapping key 'b' having the same value
    p3 = p1.union(p2)

    assert p3.get_from_full_name("a") == 1.0
    assert p3.get_from_full_name("b") == 2.0
    assert p3.get_from_full_name("c") == 3.0


def test_params_map_lower_case_key_not_found():
    """Test ParamsMap with lower_case enabled but key still not found.

    This covers the branch where lower_case is True but the lowercased
    key is not in the map.
    """
    p = ParamsMap({"my_key": 4.2})
    p.use_lower_case_keys(True)

    # Even with lower_case enabled, a completely different key should raise
    with pytest.raises(KeyError, match="Key NONEXISTENT not found"):
        _ = p.get_from_full_name("NONEXISTENT")


def test_handle_unused_params_all_used():
    """Test handle_unused_params when all parameters are used.

    This covers the early return branch when there are no unused keys.
    """
    params = ParamsMap({"a": 1.0, "b": 2.0})

    # Use all parameters
    _ = params.get_from_full_name("a")
    _ = params.get_from_full_name("b")

    # Should not raise or warn when all keys are used
    handle_unused_params(params=params, updated_records=[], raise_on_unused=False)
    handle_unused_params(params=params, updated_records=[], raise_on_unused=True)


def test_params_map_with_list_of_floats():
    """Test ParamsMap accepts a list of floats as a valid value.

    This covers the branch in _validate_params_map_value where
    the value is a list and all elements are floats.
    """
    params = ParamsMap({"a": [1.0, 2.0, 3.0]})
    assert "a" in params
