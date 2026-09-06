"""Tests for UpdatableCollection."""

import pytest

from firecrown.updatable import (
    ParamsMap,
    RequiredParameters,
    SamplerParameter,
    Updatable,
    UpdatableCollection,
    UpdatableUsageRecord,
)
from tests.updatable.updatable_test_support import MinimalUpdatable, SimpleUpdatable

# pylint: disable-msg=E1101


def test_updatable_collection_record():
    """Test record creation for UpdatableCollection."""
    coll = UpdatableCollection([SimpleUpdatable("first"), MinimalUpdatable("second")])

    params = ParamsMap({"first_x": 1.0, "first_y": 2.0, "second_a": 3.0})
    updated_records: list[UpdatableUsageRecord] = []

    coll.update(params=params, updated_record=updated_records)

    assert len(updated_records) == 2
    first_record = updated_records[0]
    assert first_record.cls == "SimpleUpdatable"
    assert first_record.prefix == "first"
    assert sorted(first_record.sampler_params) == ["x", "y"]

    second_record = updated_records[1]
    assert second_record.cls == "MinimalUpdatable"
    assert second_record.prefix == "second"
    assert second_record.sampler_params == ["a"]


def test_updatable_collection_appends():
    coll: UpdatableCollection[Updatable] = UpdatableCollection()
    assert len(coll) == 0

    coll.append(SimpleUpdatable())
    assert len(coll) == 1
    first = coll[0]
    assert isinstance(first, SimpleUpdatable)
    assert first.x is None
    assert first.y is None
    assert coll.required_parameters() == RequiredParameters(
        [
            SamplerParameter(name="x", default_value=2.0),
            SamplerParameter(name="y", default_value=3.0),
        ]
    )

    coll.append(MinimalUpdatable())
    assert len(coll) == 2
    second = coll[1]
    assert isinstance(second, MinimalUpdatable)
    assert second.a is None
    assert coll.required_parameters() == RequiredParameters(
        [
            SamplerParameter(name="x", default_value=2.0),
            SamplerParameter(name="y", default_value=3.0),
            SamplerParameter(name="a", default_value=1.0),
        ]
    )


def test_updatable_collection_updates():
    coll: UpdatableCollection[Updatable] = UpdatableCollection()
    assert len(coll) == 0

    coll.append(SimpleUpdatable())
    assert len(coll) == 1
    first = coll[0]
    assert isinstance(first, SimpleUpdatable)
    assert first.x is None
    assert first.y is None

    new_params = {"x": -1.0, "y": 5.5}
    coll.update(ParamsMap(new_params))
    assert len(coll) == 1
    assert first.x == -1.0
    assert first.y == 5.5


def test_updatable_collection_rejects_nonupdatables():
    coll: UpdatableCollection[Updatable] = UpdatableCollection()
    assert len(coll) == 0

    with pytest.raises(TypeError):
        coll.append(3)  # type: ignore[arg-type] # intentionally wrong type
    assert len(coll) == 0


def test_updatable_collection_construction():
    good_list = [SimpleUpdatable(), SimpleUpdatable()]
    good = UpdatableCollection(good_list)
    assert len(good) == 2

    bad_list = [1]
    with pytest.raises(TypeError):
        _ = UpdatableCollection(bad_list)  # pylint: disable-msg=W0612


def test_updatable_collection_insertion():
    x = UpdatableCollection([MinimalUpdatable()])
    assert len(x) == 1
    assert isinstance(x[0], MinimalUpdatable)

    x[0] = SimpleUpdatable()
    assert len(x) == 1
    assert isinstance(x[0], SimpleUpdatable)

    with pytest.raises(TypeError):
        x[0] = 1


def test_updatable_collection_is_updated():
    obj: UpdatableCollection[Updatable] = UpdatableCollection([SimpleUpdatable()])
    new_params = {"x": -1.0, "y": 5.5}

    assert not obj.is_updated()
    obj.update(ParamsMap(new_params))
    assert obj.is_updated()


def test_updatablecollection_without_derived_parameters():
    obj: UpdatableCollection[Updatable] = UpdatableCollection()

    assert obj.get_derived_parameters() is None


def test_updatablecollection_with_items_without_derived_parameters():
    """get_derived_parameters returns None when all items have no derived parameters."""
    obj: UpdatableCollection[Updatable] = UpdatableCollection()

    # Add updatables that do not implement _get_derived_parameters (return None)
    obj.append(MinimalUpdatable())
    obj.append(SimpleUpdatable())

    # Update them so they're in valid state for get_derived_parameters
    params = ParamsMap({"a": 1.0, "x": 2.0, "y": 3.0})
    obj.update(params)

    # First call returns empty collections, which get combined
    first_call = obj.get_derived_parameters()
    assert first_call is not None
    assert len(first_call) == 0  # Should be empty

    # Second call should hit the branch where has_any_derived stays False
    # because all individual updatables now return None
    # This covers the missing branch [432, 430]
    assert obj.get_derived_parameters() is None


def test_collection_update_already_updated():
    """Test calling update() twice on an UpdatableCollection.

    The second call should return early without updating items again
    (line 61 in _collection.py).
    """
    obj1 = SimpleUpdatable("obj1")
    obj2 = MinimalUpdatable("obj2")
    coll = UpdatableCollection([obj1, obj2])

    params = ParamsMap({"obj1_x": 1.0, "obj1_y": 2.0, "obj2_a": 3.0})

    # First update
    coll.update(params)
    assert coll.is_updated()
    assert obj1.x == 1.0
    assert obj1.y == 2.0
    assert obj2.a == 3.0

    # Modify params for second update
    params2 = ParamsMap({"obj1_x": 10.0, "obj1_y": 20.0, "obj2_a": 30.0})

    # Second update should be a no-op due to early return
    coll.update(params2)

    # Values should remain unchanged from first update
    assert obj1.x == 1.0
    assert obj1.y == 2.0
    assert obj2.a == 3.0


def test_collection_reset_with_items():
    """Test reset() on an UpdatableCollection with items.

    Verifies that reset() sets _updated to False and calls reset() on all
    contained items (lines 81-83 in _collection.py).
    """
    obj1 = SimpleUpdatable("obj1")
    obj2 = MinimalUpdatable("obj2")
    coll = UpdatableCollection([obj1, obj2])

    params = ParamsMap({"obj1_x": 1.0, "obj1_y": 2.0, "obj2_a": 3.0})

    # Update collection
    coll.update(params)
    assert coll.is_updated()
    assert obj1.is_updated()
    assert obj2.is_updated()
    assert obj1.x == 1.0
    assert obj2.a == 3.0

    # Reset collection
    coll.reset()

    # Collection should not be updated
    assert not coll.is_updated()

    # All items should be reset
    assert not obj1.is_updated()
    assert not obj2.is_updated()
    assert obj1.x is None
    assert obj2.a is None
