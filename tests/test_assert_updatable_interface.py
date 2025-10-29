"""Unit tests for assert_updatable_interface in updatable.py."""

import warnings
import pytest

from firecrown.updatable import (
    Updatable,
    UpdatableCollection,
    assert_updatable_interface,
)
from tests.test_updatable import SimpleUpdatable


def test_warn_override():
    BadUpdatable = type(
        "BadUpdatable", (Updatable,), {"update": lambda self, params: None}
    )
    obj = BadUpdatable()
    with pytest.warns(RuntimeWarning) as record:
        assert_updatable_interface(obj)
    assert any("update" in str(r.message) for r in record)


def test_raise_override():
    BadUpdatable = type(
        "BadUpdatable", (Updatable,), {"update": lambda self, params: None}
    )
    obj = BadUpdatable()
    with pytest.raises(TypeError):
        assert_updatable_interface(obj, raise_on_override=True)


def test_collection_items():
    BadUpdatable = type(
        "BadUpdatable", (Updatable,), {"update": lambda self, params: None}
    )
    coll = UpdatableCollection([BadUpdatable()])
    with pytest.warns(RuntimeWarning):
        assert_updatable_interface(coll)


def test_collection_override():
    """Ensure overriding UpdatableCollection methods is detected."""
    # create a subclass of UpdatableCollection that overrides 'update'
    BadCollection = type(
        "BadCollection", (UpdatableCollection,), {"update": lambda self, params: None}
    )
    coll = BadCollection()
    # Should warn about overridden methods
    with pytest.warns(RuntimeWarning):
        assert_updatable_interface(coll)


def test_recurses_attributes():
    BadUpdatable = type(
        "BadUpdatable", (Updatable,), {"update": lambda self, params: None}
    )
    parent = SimpleUpdatable()
    # attach a bad child as an attribute; __setattr__ will add it to _updatables
    parent.child = BadUpdatable()
    with pytest.warns(RuntimeWarning):
        assert_updatable_interface(parent)


def test_exact_single_method_override():
    """Assert the warning lists the single overridden method exactly."""
    BadUpdatable = type("BadUpdatable", (Updatable,), {"reset": lambda self: None})
    obj = BadUpdatable()
    with pytest.warns(RuntimeWarning) as record:
        assert_updatable_interface(obj)
    # message includes the python-list repr of overwritten methods
    assert "['reset']" in str(record[0].message)


def test_exact_multiple_method_override():
    """Assert the warning lists multiple overridden methods in the defined order."""
    impl = {
        "update": (lambda self, params: None),
        "required_parameters": (lambda self: None),
    }
    BadUpdatable = type("BadUpdatable", (Updatable,), impl)
    obj = BadUpdatable()
    with pytest.warns(RuntimeWarning) as record:
        assert_updatable_interface(obj)
    # according to the check order, update appears before required_parameters
    assert "['update', 'required_parameters']" in str(record[0].message)


def test_collection_exact_methods():
    """Ensure UpdatableCollection overrides are reported with exact methods."""
    impl = {"reset": (lambda self: None), "get_derived_parameters": (lambda self: None)}
    BadCollection = type("BadCollection", (UpdatableCollection,), impl)
    coll = BadCollection()
    with pytest.warns(RuntimeWarning) as record:
        assert_updatable_interface(coll)
    assert "['reset', 'get_derived_parameters']" in str(record[0].message)


def test_recursive_flag_prevents_recursion():
    """If recursive=False, child updatables should not be checked."""
    BadUpdatable = type(
        "BadUpdatable", (Updatable,), {"update": lambda self, params: None}
    )
    parent = SimpleUpdatable()
    parent.child = BadUpdatable()

    # When recursive is True a warning is emitted (covered elsewhere);
    # when recursive is False, no warnings should be emitted and no exception.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_updatable_interface(parent, recursive=False, raise_on_override=True)
        assert len(w) == 0


def test_collection_recursive_flag_prevents_recursion():
    """If recursive=False, items inside an UpdatableCollection should not be checked."""
    BadUpdatable = type(
        "BadUpdatable", (Updatable,), {"update": lambda self, params: None}
    )
    coll = UpdatableCollection([BadUpdatable()])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Should not raise despite raise_on_override=True because recursive=False
        assert_updatable_interface(coll, recursive=False, raise_on_override=True)
        assert len(w) == 0


def test_collection_recursive_raise_on_override():
    BadUpdatable = type(
        "BadUpdatable", (Updatable,), {"update": lambda self, params: None}
    )
    coll = UpdatableCollection([BadUpdatable()])

    with pytest.raises(TypeError):
        assert_updatable_interface(coll, recursive=True, raise_on_override=True)


def test_non_updatable_input_raises_typeerror():
    """Passing a non-Updatable and non-UpdatableCollection should raise TypeError."""
    with pytest.raises(TypeError):
        assert_updatable_interface(42)  # type: ignore
