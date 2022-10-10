import pytest
from firecrown.updatable import Updatable, UpdatableCollection
from firecrown.parameters import (
    RequiredParameters,
    ParamsMap,
    DerivedParameterCollection,
)


class Missing_update(Updatable):
    """A type that is abstract because it does not implement _update."""

    def required_parameters(self):  # pragma: no cover
        pass

    def _reset(self) -> None:
        pass

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])


class Missing_reset(Updatable):
    """A type that is abstract because it does not implement required_parameters."""

    def _update(self, params):  # pragma: no cover
        pass

    def required_parameters(self):  # pragma: no cover
        pass

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])


class Missing_required_parameters(Updatable):
    """A type that is abstract because it does not implement required_parameters."""

    def _update(self, params):  # pragma: no cover
        pass

    def _reset(self) -> None:
        pass

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])


class MinimalUpdatable(Updatable):
    """A concrete time that implements Updatable."""

    def __init__(self):
        """Initialize object with defaulted value."""
        super().__init__()

        self.a = 1.0

    def _update(self, params):
        self.a = params.get_from_prefix_param(None, "a")

    def _reset(self) -> None:
        pass

    def required_parameters(self):
        return RequiredParameters(["a"])

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])


class SimpleUpdatable(Updatable):
    """A concrete type that implements Updatable."""

    def __init__(self):
        """Initialize object with defaulted values."""
        super().__init__()

        self.x = 0.0
        self.y = 2.5

    def _update(self, params):
        self.x = params.get_from_prefix_param(None, "x")
        self.y = params.get_from_prefix_param(None, "y")

    def _reset(self) -> None:
        pass

    def required_parameters(self):
        return RequiredParameters(["x", "y"])

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])


def test_verify_abstract_interface():
    with pytest.raises(TypeError):
        x = Missing_update()  # pylint: disable-msg=E0110,W0612
    with pytest.raises(TypeError):
        x = Missing_reset()  # pylint: disable-msg=E0110,W0612
    with pytest.raises(TypeError):
        x = Missing_required_parameters()  # pylint: disable-msg=E0110,W0612


def test_simple_updatable():
    obj = SimpleUpdatable()
    expected_params = RequiredParameters(["x", "y"])
    assert obj.required_parameters() == expected_params
    assert obj.x == 0.0
    assert obj.y == 2.5
    new_params = ParamsMap({"x": -1.0, "y": 5.5})
    obj.update(new_params)
    assert obj.x == -1.0
    assert obj.y == 5.5


#  pylint: disable-msg=E1101
def test_updatable_collection_appends():
    coll = UpdatableCollection()
    assert len(coll) == 0

    coll.append(SimpleUpdatable())
    assert len(coll) == 1
    assert coll[0].x == 0.0
    assert coll[0].y == 2.5
    assert coll.required_parameters() == RequiredParameters(["x", "y"])

    coll.append(MinimalUpdatable())
    assert len(coll) == 2
    assert coll[1].a == 1.0
    assert coll.required_parameters() == RequiredParameters(["x", "y", "a"])


def test_updatable_collection_updates():
    coll = UpdatableCollection()
    assert len(coll) == 0

    coll.append(SimpleUpdatable())
    assert len(coll) == 1
    assert coll[0].x == 0.0
    assert coll[0].y == 2.5

    new_params = {"x": -1.0, "y": 5.5}
    coll.update(ParamsMap(new_params))
    assert len(coll) == 1
    assert coll[0].x == -1.0
    assert coll[0].y == 5.5


def test_updatable_collection_rejects_nonupdatables():
    coll = UpdatableCollection()
    assert len(coll) == 0

    with pytest.raises(TypeError):
        coll.append(3)  # int is not a subtype of Updatable
    assert len(coll) == 0


def test_updatable_collection_construction():
    good_list = [SimpleUpdatable(), SimpleUpdatable()]
    good = UpdatableCollection(good_list)
    assert len(good) == 2

    bad_list = [1]
    with pytest.raises(TypeError):
        x = UpdatableCollection(bad_list)  #  pylint: disable-msg=W0612


def test_updatable_collection_insertion():
    x = UpdatableCollection([MinimalUpdatable()])
    assert len(x) == 1
    assert isinstance(x[0], MinimalUpdatable)

    x[0] = SimpleUpdatable()
    assert len(x) == 1
    assert isinstance(x[0], SimpleUpdatable)

    with pytest.raises(TypeError):
        x[0] = 1
