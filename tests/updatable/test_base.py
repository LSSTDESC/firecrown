"""Tests for the Updatable base class."""

from itertools import permutations

import numpy as np
import pytest

from firecrown import updatable
from firecrown.updatable import (
    DerivedParameter,
    DerivedParameterCollection,
    ParamsMap,
    RequiredParameters,
    SamplerParameter,
    Updatable,
    UpdatableUsageRecord,
)
from tests.updatable.updatable_test_support import (
    MinimalUpdatable,
    SimpleUpdatable,
    UpdatableWithDerived,
)

# pylint: disable-msg=E1101


def test_get_params_names():
    obj = SimpleUpdatable()
    found_names = obj.get_params_names()
    assert set(found_names) == {"x", "y"}


def test_simple_updatable():
    obj = SimpleUpdatable()
    expected_params = RequiredParameters(
        [
            SamplerParameter(name="y", default_value=3.0),
            SamplerParameter(name="x", default_value=2.0),
        ]
    )
    assert obj.required_parameters() == expected_params
    found_names = obj.get_params_names()
    assert "x" in found_names
    assert "y" in found_names
    assert obj.x is None
    assert obj.y is None
    assert not obj.is_updated()
    new_params = ParamsMap({"x": -1.0, "y": 5.5})
    obj.update(new_params)
    assert obj.x == -1.0
    assert obj.y == 5.5
    assert obj.is_updated()


def test_set_sampler_parameter():
    my_updatable = MinimalUpdatable()
    my_param = updatable.register_new_updatable_parameter(default_value=42.0)
    my_param.set_fullname(prefix=None, name="the_meaning_of_life")
    my_updatable.set_sampler_parameter(my_param)

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life is None


def test_set_sampler_parameter_rejects_internal_parameter():
    my_updatable = MinimalUpdatable()
    my_param = updatable.register_new_updatable_parameter(
        value=42.0, default_value=41.0
    )

    with pytest.raises(TypeError):
        my_updatable.set_sampler_parameter(my_param)  # type: ignore[arg-type]


def test_set_sampler_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    my_param = updatable.register_new_updatable_parameter(default_value=42.0)
    my_param.set_fullname(prefix=None, name="the_meaning_of_life")
    my_param_same = updatable.register_new_updatable_parameter(default_value=42.0)
    my_param_same.set_fullname(prefix=None, name="the_meaning_of_life")

    my_updatable.set_sampler_parameter(my_param)

    with pytest.raises(ValueError):
        my_updatable.set_sampler_parameter(my_param_same)


def test_set_internal_parameter():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life",
        updatable.register_new_updatable_parameter(value=1.0, default_value=42.0),
    )

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life == 1.0


def test_set_parameter_using_internal_parameter():
    my_updatable = MinimalUpdatable()
    ip = updatable.InternalParameter(2112)
    my_updatable.set_parameter("epic_Rush_album", ip)

    assert hasattr(my_updatable, "epic_Rush_album")
    assert my_updatable.epic_Rush_album == 2112


def test_set_internal_parameter_rejects_sampler_parameter():
    my_updatable = MinimalUpdatable()
    with pytest.raises(TypeError):
        my_updatable.set_internal_parameter(
            "sampler_param",
            updatable.register_new_updatable_parameter(  # type: ignore[arg-type]
                default_value=1.0
            ),
        )


def test_set_internal_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life",
        updatable.register_new_updatable_parameter(value=1.0, default_value=42.0),
    )

    with pytest.raises(ValueError):
        my_updatable.set_internal_parameter(
            "the_meaning_of_life",
            updatable.register_new_updatable_parameter(value=1.0, default_value=42.0),
        )


def test_set_parameter():
    my_updatable = MinimalUpdatable()
    my_updatable.set_parameter(
        "the_meaning_of_life",
        updatable.register_new_updatable_parameter(value=1.0, default_value=42.0),
    )
    my_updatable.set_parameter(
        "no_meaning_of_life",
        updatable.register_new_updatable_parameter(default_value=42.0),
    )

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life == 1.0

    assert hasattr(my_updatable, "no_meaning_of_life")
    assert my_updatable.no_meaning_of_life is None


def test_update_rejects_internal_parameters():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life",
        updatable.register_new_updatable_parameter(value=2.0, default_value=42.0),
    )
    assert hasattr(my_updatable, "the_meaning_of_life")

    params = ParamsMap({"a": 1.1, "the_meaning_of_life": 34.0})
    with pytest.raises(
        TypeError,
        match="Items of type InternalParameter cannot be modified through update",
    ):
        my_updatable.update(params)

    assert my_updatable.a is None
    assert my_updatable.the_meaning_of_life == 2.0


def test_setattr_with_list_of_updatables():
    """Test setting an attribute to a list of Updatable objects.

    This tests the code path in __setattr__ that handles iterables containing
    UpdatableProtocol instances (line 69 in _base.py).
    """
    parent = SimpleUpdatable("parent")
    child1 = MinimalUpdatable("child1")
    child2 = MinimalUpdatable("child2")

    # Set attribute to a list of updatables
    parent.children = [child1, child2]  # pylint: disable=attribute-defined-outside-init

    # Verify both children were added to _updatables
    assert child1 in parent._updatables  # pylint: disable=protected-access
    assert child2 in parent._updatables  # pylint: disable=protected-access

    # Verify they can be updated through the parent
    params = ParamsMap(
        {"parent_x": 1.0, "parent_y": 2.0, "child1_a": 3.0, "child2_a": 4.0}
    )
    parent.update(params)
    assert parent.x == 1.0
    assert parent.y == 2.0
    assert child1.a == 3.0
    assert child2.a == 4.0


def test_update_already_updated_with_updated_record():
    """Test calling update() twice with updated_record parameter.

    When an object is already updated and update() is called again with
    updated_record tracking, it should add a record with already_updated=True
    and return early (lines 153-165 in _base.py).
    """
    obj = SimpleUpdatable("test")
    params = ParamsMap({"test_x": 1.0, "test_y": 2.0})

    # First update
    updated_records: list[UpdatableUsageRecord] = []
    obj.update(params, updated_record=updated_records)
    assert len(updated_records) == 1
    assert not updated_records[0].already_updated

    # Second update with tracking - should record already_updated=True
    updated_records2: list[UpdatableUsageRecord] = []
    obj.update(params, updated_record=updated_records2)
    assert len(updated_records2) == 1
    assert updated_records2[0].already_updated is True
    assert updated_records2[0].cls == "SimpleUpdatable"
    assert updated_records2[0].prefix == "test"
    assert updated_records2[0].sampler_params == []
    assert updated_records2[0].internal_params == []


def test_update_already_updated_without_record():
    """Test calling update() twice without updated_record parameter.

    When an object is already updated and update() is called again without
    updated_record tracking, it should return early without recording anything
    (branch 153->165 in _base.py).
    """
    obj = SimpleUpdatable("test")
    params = ParamsMap({"test_x": 1.0, "test_y": 2.0})

    # First update
    obj.update(params)
    assert obj.is_updated()
    assert obj.x == 1.0
    assert obj.y == 2.0

    # Modify params
    params2 = ParamsMap({"test_x": 10.0, "test_y": 20.0})

    # Second update without record tracking - should be a no-op
    obj.update(params2)

    # Values should remain unchanged
    assert obj.x == 1.0
    assert obj.y == 2.0


def test_reset_when_not_updated():
    """Test calling reset() on an object that hasn't been updated.

    Should return early without doing anything (lines 234-235 in _base.py).
    """
    obj = SimpleUpdatable("test")

    # Object hasn't been updated yet
    assert not obj.is_updated()

    # Call reset - should be a no-op
    obj.reset()

    # Still not updated
    assert not obj.is_updated()

    # Parameters should still be None (not set)
    assert obj.x is None
    assert obj.y is None


def test_reset_with_nested_updatables_and_sampler_params():
    """Test reset() with nested updatables and sampler parameters.

    Verifies that reset():
    1. Resets nested updatables
    2. Sets sampler parameters back to None
    3. Clears the _updated flag
    (lines 239-248 in _base.py)
    """
    parent = SimpleUpdatable("parent")
    child = MinimalUpdatable("child")
    parent.child = child  # pylint: disable=attribute-defined-outside-init

    params = ParamsMap({"parent_x": 1.0, "parent_y": 2.0, "child_a": 3.0})

    # Update both parent and child
    parent.update(params)
    assert parent.is_updated()
    assert child.is_updated()
    assert parent.x == 1.0
    assert parent.y == 2.0
    assert child.a == 3.0

    # Reset parent (should cascade to child)
    parent.reset()

    # Parent should be reset
    assert not parent.is_updated()
    assert parent.x is None
    assert parent.y is None

    # Child should also be reset
    assert not child.is_updated()
    assert child.a is None


def test_tuple_attribute_adds_to_updatables():
    """Test that setting a tuple attribute adds Updatable elements to _updatables.

    When an Updatable container has an attribute set to an iterable containing
    Updatable objects, those objects should be added to the container's
    _updatables list so they are updated when the container is updated.
    """

    container = Updatable()
    item1 = MinimalUpdatable()
    item2 = MinimalUpdatable()

    # Setting a tuple attribute should add the items to _updatables
    container.test_tuple = (item1, item2)

    # Items should be added to _updatables
    assert item1 in container._updatables  # pylint: disable=protected-access
    assert item2 in container._updatables  # pylint: disable=protected-access


@pytest.fixture(name="nested_updatables", params=permutations(range(3)))
def fixture_nested_updatables(request):
    updatables = np.array(
        [MinimalUpdatable(), SimpleUpdatable(), UpdatableWithDerived()]
    )

    # Reorder the updatables and set up the nesting
    updatables = updatables[list(request.param)]
    updatables[0].sub_updatable = updatables[1]
    updatables[1].sub_updatable = updatables[2]

    return updatables


def test_nesting_updatables_missing_parameters(nested_updatables):
    base = nested_updatables[0]
    assert isinstance(base, Updatable)

    params = ParamsMap({})

    with pytest.raises(
        RuntimeError,
    ):
        base.update(params)

    params = ParamsMap({"a": 1.1})

    with pytest.raises(
        RuntimeError,
    ):
        base.update(params)

    params = ParamsMap({"a": 1.1, "x": 2.0, "y": 3.0})

    with pytest.raises(
        RuntimeError,
    ):
        base.update(params)

    params = ParamsMap({"a": 1.1, "x": 2.0, "y": 3.0, "A": 4.0, "B": 5.0})

    base.update(params)

    for my_updatable in nested_updatables:
        assert my_updatable.is_updated()


def test_nesting_updatables_required_parameters(nested_updatables):
    base = nested_updatables[0]
    assert isinstance(base, Updatable)

    assert base.required_parameters() == RequiredParameters(
        [
            SamplerParameter(name="a", default_value=1.0),
            SamplerParameter(name="x", default_value=2.0),
            SamplerParameter(name="y", default_value=3.0),
            SamplerParameter(name="A", default_value=2.0),
            SamplerParameter(name="B", default_value=1.0),
        ]
    )


def test_nesting_updatables_derived_parameters(nested_updatables):
    base = nested_updatables[0]
    assert isinstance(base, Updatable)

    with pytest.raises(
        RuntimeError,
        match="Derived parameters can only be obtained after update has been called.",
    ):
        base.get_derived_parameters()

    params = ParamsMap({"a": 1.1, "x": 2.0, "y": 3.0, "A": 4.0, "B": 5.0})

    base.update(params)

    derived_scale = DerivedParameter("Section", "Name", 9.0)
    derived_parameters = DerivedParameterCollection([derived_scale])

    assert base.get_derived_parameters() == derived_parameters
    assert base.get_derived_parameters() is None
