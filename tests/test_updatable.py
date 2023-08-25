"""
Tests for the Updatable class.
"""
import pytest
from firecrown.updatable import Updatable, UpdatableCollection
from firecrown import parameters
from firecrown.parameters import (
    RequiredParameters,
    ParamsMap,
    DerivedParameterCollection,
)


class MinimalUpdatable(Updatable):
    """A concrete time that implements Updatable."""

    def __init__(self):
        """Initialize object with defaulted value."""
        super().__init__()

        self.a = parameters.create()

    def _update(self, params):
        pass

    def _reset(self) -> None:
        pass

    def _required_parameters(self):
        return RequiredParameters([])

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])


class SimpleUpdatable(Updatable):
    """A concrete type that implements Updatable."""

    def __init__(self):
        """Initialize object with defaulted values."""
        super().__init__()

        self.x = parameters.create()
        self.y = parameters.create()

    def _update(self, params):
        pass

    def _reset(self) -> None:
        pass

    def _required_parameters(self):
        return RequiredParameters([])

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])


def test_simple_updatable():
    obj = SimpleUpdatable()
    expected_params = RequiredParameters(["x", "y"])
    assert obj.required_parameters() == expected_params
    assert obj.x is None
    assert obj.y is None
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
    assert coll[0].x is None
    assert coll[0].y is None
    assert coll.required_parameters() == RequiredParameters(["x", "y"])

    coll.append(MinimalUpdatable())
    assert len(coll) == 2
    assert coll[1].a is None
    assert coll.required_parameters() == RequiredParameters(["x", "y", "a"])


def test_updatable_collection_updates():
    coll = UpdatableCollection()
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
    coll = UpdatableCollection()
    assert len(coll) == 0

    with pytest.raises(TypeError):
        coll.append(3)  # type: ignore
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
    my_updatable.set_sampler_parameter("the_meaning_of_life", parameters.create())

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life is None


def test_set_sampler_parameter_rejects_internal_parameter():
    my_updatable = MinimalUpdatable()

    with pytest.raises(TypeError):
        my_updatable.set_sampler_parameter(
            "the_meaning_of_life", parameters.create(42.0)
        )


def test_set_sampler_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    my_updatable.set_sampler_parameter("the_meaning_of_life", parameters.create())

    with pytest.raises(ValueError):
        my_updatable.set_sampler_parameter("the_meaning_of_life", parameters.create())


def test_set_internal_parameter():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter("the_meaning_of_life", parameters.create(42.0))

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life == 42.0


def test_set_internal_parameter_rejects_sampler_parameter():
    my_updatable = MinimalUpdatable()
    with pytest.raises(TypeError):
        my_updatable.set_internal_parameter("sampler_param", parameters.create())


def test_set_internal_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter("the_meaning_of_life", parameters.create(42.0))

    with pytest.raises(ValueError):
        my_updatable.set_internal_parameter(
            "the_meaning_of_life", parameters.create(42.0)
        )

    with pytest.raises(ValueError):
        my_updatable.set_internal_parameter(
            "the_meaning_of_life", parameters.create(41.0)
        )


def test_update_rejects_internal_parameters():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter("the_meaning_of_life", parameters.create(42.0))
    assert hasattr(my_updatable, "the_meaning_of_life")

    params = ParamsMap({"a": 1.1, "the_meaning_of_life": 34.0})
    with pytest.raises(
        TypeError,
        match="Items of type InternalParameter cannot be modified through update",
    ):
        my_updatable.update(params)

    assert my_updatable.a is None
    assert my_updatable.the_meaning_of_life == 42.0
