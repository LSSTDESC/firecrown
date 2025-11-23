"""
Tests for the Updatable class.
"""

from itertools import permutations
import pytest
import numpy as np

from firecrown.updatable import (
    Updatable,
    UpdatableCollection,
    UpdatableUsageRecord,
    get_default_params,
    get_default_params_map,
)
from firecrown import parameters
from firecrown.parameters import (
    RequiredParameters,
    ParamsMap,
    DerivedParameter,
    DerivedParameterCollection,
    SamplerParameter,
)


class MinimalUpdatable(Updatable):
    """A concrete time that implements Updatable."""

    def __init__(self, prefix: str | None = None):
        """Initialize object with defaulted value."""
        super().__init__(prefix)
        self.a = parameters.register_new_updatable_parameter(default_value=1.0)


class SimpleUpdatable(Updatable):  # pylint: disable=too-many-instance-attributes
    """A concrete type that implements Updatable."""

    def __init__(self, prefix: str | None = None):
        """Initialize object with defaulted values."""
        super().__init__(prefix)

        self.x = parameters.register_new_updatable_parameter(default_value=2.0)
        self.y = parameters.register_new_updatable_parameter(default_value=3.0)


class UpdatableWithDerived(Updatable):
    """A concrete type that implements Updatable that implements derived parameters."""

    def __init__(self):
        """Initialize object with defaulted values."""
        super().__init__()

        self.A = parameters.register_new_updatable_parameter(default_value=2.0)
        self.B = parameters.register_new_updatable_parameter(default_value=1.0)

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_scale = DerivedParameter("Section", "Name", self.A + self.B)
        derived_parameters = DerivedParameterCollection([derived_scale])

        return derived_parameters


def test_updatable_reports():
    su = SimpleUpdatable("bob")
    mu = MinimalUpdatable("larry")
    su.mu = mu  # pylint: disable=attribute-defined-outside-init

    params = ParamsMap({"bob_x": 1.0, "bob_y": 2.0, "larry_a": 3.0})
    updated_records: list[UpdatableUsageRecord] = []
    su.update(params=params, updated_record=updated_records)
    lines = []
    for updated_record in updated_records:
        lines += updated_record.get_log_lines()
    after_use = "\n".join(lines)
    assert "SimpleUpdatable(bob)" in after_use
    assert "Sampler parameters used:  ['x', 'y']" in after_use
    assert "MinimalUpdatable(larry)" in after_use
    assert "Sampler parameters used:  ['a']" in after_use


def test_updatable_record_single():
    """Test record creation for a single Updatable without nesting."""
    obj = SimpleUpdatable("test")
    params = ParamsMap({"test_x": 1.0, "test_y": 2.0})
    updated_records: list[UpdatableUsageRecord] = []

    obj.update(params=params, updated_record=updated_records)

    assert len(updated_records) == 1
    record = updated_records[0]
    assert record.cls == "SimpleUpdatable"
    assert record.prefix == "test"
    assert sorted(record.sampler_params) == ["x", "y"]
    assert len(record.internal_params) == 0
    assert len(record.child_records) == 0


def test_updatable_record_with_internal_params():
    """Test record creation with both sampler and internal parameters."""
    obj = SimpleUpdatable("test")
    obj.set_internal_parameter(
        "internal1",
        parameters.register_new_updatable_parameter(value=1.0, default_value=1.0),
    )
    obj.set_internal_parameter(
        "internal2",
        parameters.register_new_updatable_parameter(value=2.0, default_value=2.0),
    )

    params = ParamsMap({"test_x": 1.0, "test_y": 2.0})
    updated_records: list[UpdatableUsageRecord] = []

    obj.update(params=params, updated_record=updated_records)

    assert len(updated_records) == 1
    record = updated_records[0]
    assert sorted(record.sampler_params) == ["x", "y"]
    assert sorted(record.internal_params) == ["internal1", "internal2"]


def test_updatable_record_nested():
    """Test record creation for nested Updatable objects."""
    parent = SimpleUpdatable("parent")
    child = MinimalUpdatable("child")
    parent.nested = child  # pylint: disable=attribute-defined-outside-init

    params = ParamsMap({"parent_x": 1.0, "parent_y": 2.0, "child_a": 3.0})
    updated_records: list[UpdatableUsageRecord] = []

    parent.update(params=params, updated_record=updated_records)

    assert len(updated_records) == 1
    parent_record = updated_records[0]
    assert parent_record.cls == "SimpleUpdatable"
    assert parent_record.prefix == "parent"
    assert sorted(parent_record.sampler_params) == ["x", "y"]

    assert len(parent_record.child_records) == 1
    child_record = parent_record.child_records[0]
    assert child_record.cls == "MinimalUpdatable"
    assert child_record.prefix == "child"
    assert child_record.sampler_params == ["a"]


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


def test_updatable_record_complex_hierarchy():
    """Test record creation for a complex hierarchy of nested objects."""
    root = SimpleUpdatable("root")
    branch1 = MinimalUpdatable("branch1")
    branch2 = SimpleUpdatable("branch2")
    leaf1 = MinimalUpdatable("leaf1")
    leaf2 = UpdatableWithDerived()

    # Create hierarchy:
    # root -> branch1 -> leaf1
    #      -> branch2 -> leaf2
    # pylint: disable=attribute-defined-outside-init
    root.b1 = branch1
    root.b2 = branch2
    branch1.leaf = leaf1
    branch2.leaf = leaf2
    # pylint: enable=attribute-defined-outside-init

    params = ParamsMap(
        {
            "root_x": 1.0,
            "root_y": 2.0,
            "branch1_a": 3.0,
            "branch2_x": 4.0,
            "branch2_y": 5.0,
            "leaf1_a": 6.0,
            "A": 7.0,
            "B": 8.0,
        }
    )
    updated_records: list[UpdatableUsageRecord] = []

    root.update(params=params, updated_record=updated_records)

    assert len(updated_records) == 1
    root_record = updated_records[0]
    assert root_record.cls == "SimpleUpdatable"
    assert root_record.prefix == "root"
    assert len(root_record.child_records) == 2

    # Verify the complete hierarchy is captured in records
    branch1_record = next(r for r in root_record.child_records if r.prefix == "branch1")
    assert len(branch1_record.child_records) == 1
    assert branch1_record.child_records[0].prefix == "leaf1"

    branch2_record = next(r for r in root_record.child_records if r.prefix == "branch2")
    assert len(branch2_record.child_records) == 1
    leaf2_record = branch2_record.child_records[0]
    assert leaf2_record.cls == "UpdatableWithDerived"


def test_updatable_record_empty_params():
    """Test record creation for an object with no parameters."""

    class EmptyUpdatable(Updatable):
        """An Updatable with no parameters."""

    obj = EmptyUpdatable("empty")
    updated_records: list[UpdatableUsageRecord] = []
    obj.update(ParamsMap({}), updated_record=updated_records)

    assert len(updated_records) == 1
    record = updated_records[0]
    assert record.cls == "EmptyUpdatable"
    assert record.prefix == "empty"
    assert len(record.sampler_params) == 0
    assert len(record.internal_params) == 0


def test_get_params_names():
    obj = SimpleUpdatable()
    found_names = obj.get_params_names()
    assert set(found_names) == set(["x", "y"])


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


#  pylint: disable-msg=E1101
def test_updatable_collection_appends():
    coll: UpdatableCollection = UpdatableCollection()
    assert len(coll) == 0

    coll.append(SimpleUpdatable())
    assert len(coll) == 1
    assert coll[0].x is None
    assert coll[0].y is None
    assert coll.required_parameters() == RequiredParameters(
        [
            SamplerParameter(name="x", default_value=2.0),
            SamplerParameter(name="y", default_value=3.0),
        ]
    )

    coll.append(MinimalUpdatable())
    assert len(coll) == 2
    assert coll[1].a is None
    assert coll.required_parameters() == RequiredParameters(
        [
            SamplerParameter(name="x", default_value=2.0),
            SamplerParameter(name="y", default_value=3.0),
            SamplerParameter(name="a", default_value=1.0),
        ]
    )


def test_updatable_collection_updates():
    coll: UpdatableCollection = UpdatableCollection()
    assert len(coll) == 0

    coll.append(SimpleUpdatable())
    assert len(coll) == 1
    assert coll[0].x is None
    assert coll[0].y is None

    new_params = {"x": -1.0, "y": 5.5}
    coll.update(ParamsMap(new_params))
    assert len(coll) == 1
    assert coll[0].x == -1.0
    assert coll[0].y == 5.5


def test_updatable_collection_rejects_nonupdatables():
    coll: UpdatableCollection = UpdatableCollection()
    assert len(coll) == 0

    with pytest.raises(TypeError):
        coll.append(3)
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


def test_set_sampler_parameter():
    my_updatable = MinimalUpdatable()
    my_param = parameters.register_new_updatable_parameter(default_value=42.0)
    my_param.set_fullname(prefix=None, name="the_meaning_of_life")
    my_updatable.set_sampler_parameter(my_param)

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life is None


def test_set_sampler_parameter_rejects_internal_parameter():
    my_updatable = MinimalUpdatable()
    my_param = parameters.register_new_updatable_parameter(
        value=42.0, default_value=41.0
    )

    with pytest.raises(TypeError):
        my_updatable.set_sampler_parameter(my_param)


def test_set_sampler_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    my_param = parameters.register_new_updatable_parameter(default_value=42.0)
    my_param.set_fullname(prefix=None, name="the_meaning_of_life")
    my_param_same = parameters.register_new_updatable_parameter(default_value=42.0)
    my_param_same.set_fullname(prefix=None, name="the_meaning_of_life")

    my_updatable.set_sampler_parameter(my_param)

    with pytest.raises(ValueError):
        my_updatable.set_sampler_parameter(my_param_same)


def test_set_internal_parameter():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life",
        parameters.register_new_updatable_parameter(value=1.0, default_value=42.0),
    )

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life == 1.0


def test_set_parameter_using_internal_parameter():
    my_updatable = MinimalUpdatable()
    ip = parameters.InternalParameter(2112)
    my_updatable.set_parameter("epic_Rush_album", ip)

    assert hasattr(my_updatable, "epic_Rush_album")
    assert my_updatable.epic_Rush_album == 2112


def test_set_internal_parameter_rejects_sampler_parameter():
    my_updatable = MinimalUpdatable()
    with pytest.raises(TypeError):
        my_updatable.set_internal_parameter(
            "sampler_param",
            parameters.register_new_updatable_parameter(default_value=1.0),
        )


def test_set_internal_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life",
        parameters.register_new_updatable_parameter(value=1.0, default_value=42.0),
    )

    with pytest.raises(ValueError):
        my_updatable.set_internal_parameter(
            "the_meaning_of_life",
            parameters.register_new_updatable_parameter(value=1.0, default_value=42.0),
        )


def test_set_parameter():
    my_updatable = MinimalUpdatable()
    my_updatable.set_parameter(
        "the_meaning_of_life",
        parameters.register_new_updatable_parameter(value=1.0, default_value=42.0),
    )
    my_updatable.set_parameter(
        "no_meaning_of_life",
        parameters.register_new_updatable_parameter(default_value=42.0),
    )

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life == 1.0

    assert hasattr(my_updatable, "no_meaning_of_life")
    assert my_updatable.no_meaning_of_life is None


def test_update_rejects_internal_parameters():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life",
        parameters.register_new_updatable_parameter(value=2.0, default_value=42.0),
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


def test_updatable_collection_is_updated():
    obj: UpdatableCollection = UpdatableCollection([SimpleUpdatable()])
    new_params = {"x": -1.0, "y": 5.5}

    assert not obj.is_updated()
    obj.update(ParamsMap(new_params))
    assert obj.is_updated()


def test_updatablecollection_without_derived_parameters():
    obj: UpdatableCollection = UpdatableCollection()

    assert obj.get_derived_parameters() is None


def test_updatablecollection_with_items_without_derived_parameters():
    """get_derived_parameters returns None when all items have no derived parameters."""
    obj: UpdatableCollection = UpdatableCollection()

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

    for updatable in nested_updatables:
        assert updatable.is_updated()


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


# Tests for UpdatableUsageRecord (moved from test_parameters.py)


def test_updatable_usage_record_empty_and_print_empty():
    rec = UpdatableUsageRecord(
        cls="EmptyUpdatable",
        prefix="pfx",
        obj_id=1,
        sampler_params=[],
        internal_params=[],
        child_records=[],
    )

    assert rec.is_empty
    assert not rec.is_empty_parent

    # by default empty records are omitted
    assert rec.get_log_lines() == []

    # but when print_empty=True a header line is produced
    lines = rec.get_log_lines(print_empty=True)
    assert lines == ["EmptyUpdatable(pfx): "]


def test_updatable_usage_record_parent_collapses_and_printing():
    child = UpdatableUsageRecord(
        cls="Child",
        prefix="c",
        obj_id=2,
        sampler_params=["x"],
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

    # Because parent has no params but exactly one child, the parent's record
    # should collapse into the child lines, with the parent included as a prefix
    lines = parent.get_log_lines()
    assert lines == [
        "Parent(p) => Child(c): ",
        "  Sampler parameters used:  ['x']",
    ]


def test_updatable_usage_record_indent_and_child_recursion():
    child = UpdatableUsageRecord(
        cls="Child",
        prefix="c",
        obj_id=2,
        sampler_params=["b"],
        internal_params=[],
        child_records=[],
    )
    parent = UpdatableUsageRecord(
        cls="Parent",
        prefix="p",
        obj_id=1,
        sampler_params=["a"],
        internal_params=["i"],
        child_records=[child],
    )

    lines = parent.get_log_lines()

    expected = [
        "Parent(p): ",
        "  Sampler parameters used:  ['a']",
        "  Internal parameters used: ['i']",
        "  Child(c): ",
        "    Sampler parameters used:  ['b']",
    ]

    assert lines == expected


def test_updatable_usage_record_empty_child_and_print_empty_options():
    # child is empty
    child = UpdatableUsageRecord(
        cls="Child",
        prefix="c",
        obj_id=3,
        sampler_params=[],
        internal_params=[],
        child_records=[],
    )
    parent = UpdatableUsageRecord(
        cls="Parent",
        prefix="p",
        obj_id=4,
        sampler_params=[],
        internal_params=[],
        child_records=[child],
    )

    # with default print_empty=False, collapse should result in empty output
    assert parent.get_log_lines() == []

    # with print_empty=True we should see the collapsed header
    lines = parent.get_log_lines(print_empty=True)
    assert lines == ["Parent(p) => Child(c): "]


def test_updatable_usage_record_already_updated_flag():
    """If an UpdatableUsageRecord indicates it was already updated, the
    get_log_lines should return a single line noting it was already updated.
    """
    rec = UpdatableUsageRecord(
        cls="Already",
        prefix=None,
        obj_id=1,
        sampler_params=[],
        internal_params=[],
        child_records=[],
        already_updated=True,
    )

    lines = rec.get_log_lines(print_empty=True)
    assert lines == ["Already: (already updated)"]


def test_updatable_usage_record_internal_params_only():
    """If sampler_params is empty but internal_params is non-empty,
    is_empty should be False and get_log_lines should show the internal
    parameters line (this covers the second early-return branch).
    """
    rec = UpdatableUsageRecord(
        cls="OnlyInternal",
        prefix="p",
        obj_id=7,
        sampler_params=[],
        internal_params=["i"],
        child_records=[],
    )

    assert not rec.is_empty
    lines = rec.get_log_lines()
    assert lines == ["OnlyInternal(p): ", "  Internal parameters used: ['i']"]


# Tests for _base.py edge cases to achieve 100% coverage


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


# Tests for _collection.py edge cases to achieve 100% coverage


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


# Tests for _utils.py edge cases to achieve 100% coverage


def test_get_default_params_empty_args():
    """Test get_default_params() with no arguments.

    Should return an empty dictionary (lines 19-23 in _utils.py).
    """
    result = get_default_params()
    assert result == {}
    assert isinstance(result, dict)


def test_get_default_params_map_empty_args():
    """Test get_default_params_map() with no arguments.

    Should return an empty ParamsMap (lines 32-33 in _utils.py).
    """
    result = get_default_params_map()
    assert isinstance(result, ParamsMap)
    assert len(result.keys()) == 0


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


def test_get_default_params_with_multiple_updatables():
    """Test get_default_params() with multiple updatables.

    Verifies that all default values are collected correctly.
    """
    obj1 = SimpleUpdatable()
    obj2 = MinimalUpdatable()

    # Test get_default_params
    result = get_default_params(obj1, obj2)
    assert result == {"x": 2.0, "y": 3.0, "a": 1.0}

    # Test get_default_params_map
    params_map = get_default_params_map(obj1, obj2)
    assert isinstance(params_map, ParamsMap)
    assert params_map.get("x") == 2.0
    assert params_map.get("y") == 3.0
    assert params_map.get("a") == 1.0
