import pytest
from firecrown.updatable import Updatable, UpdatableCollection
from firecrown.parameters import RequiredParameters, ParamsMap


class Missing_update(Updatable):
    """A type that is abstract because it does not implement _update."""

    def required_parameters(self):
        return RequiredParameters([])


class Missing_required_parameters(Updatable):
    """A type that is abstract because it does not implement required_parameters."""

    def _update(self, params):
        super()._update(params)


class MinimalUpdatable(Updatable):
    """A concrete type that implements Updatable."""

    def __init__(self):
        """Initialize objec with defaulted values."""
        self.x = 0.0
        self.y = 2.5

    def _update(self, params):
        self.x = params.get_from_prefix_param(None, "x")
        self.y = params.get_from_prefix_param(None, "y")

    def required_parameters(self):
        return RequiredParameters(["x", "y"])

    def __eq__(self, other):
        if not isinstance(other, MinimalUpdatable):
            return false
        return self.x == other.x and self.y == other.y


def test_verify_abstract_interface():
    with pytest.raises(TypeError):
        x = Missing_update()
    with pytest.raises(TypeError):
        x = Missing_required_parameters()


def test_minimal_updatable():
    obj = MinimalUpdatable()
    expected_params = RequiredParameters(["x", "y"])
    assert obj.required_parameters() == expected_params
    assert obj.x == 0.0
    assert obj.y == 2.5
    new_params = ParamsMap({"x": -1.0, "y": 5.5})
    obj.update(new_params)
    assert obj.x == -1.0
    assert obj.y == 5.5


def test_updateable_collection():
    coll = UpdatableCollection()
    coll.append(MinimalUpdatable())
    assert len(coll) == 1
    assert coll[0] == MinimalUpdatable()
    new_params = ParamsMap({"x": -1.0, "y": 5.5})

    coll.update(new_params)
