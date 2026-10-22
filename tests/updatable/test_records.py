"""Tests for UpdatableUsageRecord."""

from firecrown import updatable
from firecrown.updatable import ParamsMap, Updatable, UpdatableUsageRecord
from tests.updatable.updatable_test_support import (
    MinimalUpdatable,
    SimpleUpdatable,
    UpdatableWithDerived,
)


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
        updatable.register_new_updatable_parameter(value=1.0, default_value=1.0),
    )
    obj.set_internal_parameter(
        "internal2",
        updatable.register_new_updatable_parameter(value=2.0, default_value=2.0),
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
