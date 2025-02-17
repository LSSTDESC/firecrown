"""
Tests for the Updatable class.
"""

from itertools import permutations
import pytest
import numpy as np

from firecrown.updatable import Updatable, UpdatableCollection
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

    def __init__(self):
        """Initialize object with defaulted value."""
        super().__init__()

        with pytest.deprecated_call():
            self.a = parameters.register_new_updatable_parameter()


class SimpleUpdatable(Updatable):
    """A concrete type that implements Updatable."""

    def __init__(self):
        """Initialize object with defaulted values."""
        super().__init__()

        with pytest.deprecated_call():
            self.x = parameters.register_new_updatable_parameter()
            self.y = parameters.register_new_updatable_parameter()


class UpdatableWithDerived(Updatable):
    """A concrete type that implements Updatable that implements derived parameters."""

    def __init__(self):
        """Initialize object with defaulted values."""
        super().__init__()

        with pytest.deprecated_call():
            self.A = parameters.register_new_updatable_parameter()
            self.B = parameters.register_new_updatable_parameter()

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_scale = DerivedParameter("Section", "Name", self.A + self.B)
        derived_parameters = DerivedParameterCollection([derived_scale])

        return derived_parameters


def test_get_params_names():
    obj = SimpleUpdatable()
    found_names = obj.get_params_names()
    assert set(found_names) == set(["x", "y"])


def test_simple_updatable():
    obj = SimpleUpdatable()
    with pytest.deprecated_call():
        expected_params = RequiredParameters(
            [SamplerParameter(name="y"), SamplerParameter(name="x")]
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
    with pytest.deprecated_call():
        assert coll.required_parameters() == RequiredParameters(
            [SamplerParameter(name="x"), SamplerParameter(name="y")]
        )

    coll.append(MinimalUpdatable())
    assert len(coll) == 2
    assert coll[1].a is None
    with pytest.deprecated_call():
        assert coll.required_parameters() == RequiredParameters(
            [
                SamplerParameter(name="x"),
                SamplerParameter(name="y"),
                SamplerParameter(name="a"),
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
    with pytest.deprecated_call():
        my_param = parameters.register_new_updatable_parameter()
    my_param.set_fullname(prefix=None, name="the_meaning_of_life")
    my_updatable.set_sampler_parameter(my_param)

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life is None


def test_set_sampler_parameter_rejects_internal_parameter():
    my_updatable = MinimalUpdatable()
    my_param = parameters.register_new_updatable_parameter(42.0)

    with pytest.raises(TypeError):
        my_updatable.set_sampler_parameter(my_param)


def test_set_sampler_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    with pytest.deprecated_call():
        my_param = parameters.register_new_updatable_parameter()
    my_param.set_fullname(prefix=None, name="the_meaning_of_life")
    with pytest.deprecated_call():
        my_param_same = parameters.register_new_updatable_parameter()
    my_param_same.set_fullname(prefix=None, name="the_meaning_of_life")

    my_updatable.set_sampler_parameter(my_param)

    with pytest.raises(ValueError):
        my_updatable.set_sampler_parameter(my_param_same)


def test_set_internal_parameter():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life", parameters.register_new_updatable_parameter(42.0)
    )

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life == 42.0


def test_set_parameter_using_internal_parameter():
    my_updatable = MinimalUpdatable()
    ip = parameters.InternalParameter(2112)
    my_updatable.set_parameter("epic_Rush_album", ip)

    assert hasattr(my_updatable, "epic_Rush_album")
    assert my_updatable.epic_Rush_album == 2112


def test_set_internal_parameter_rejects_sampler_parameter():
    my_updatable = MinimalUpdatable()
    with pytest.raises(TypeError):
        with pytest.deprecated_call():
            my_updatable.set_internal_parameter(
                "sampler_param", parameters.register_new_updatable_parameter()
            )


def test_set_internal_parameter_rejects_duplicates():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life", parameters.register_new_updatable_parameter(42.0)
    )

    with pytest.raises(ValueError):
        my_updatable.set_internal_parameter(
            "the_meaning_of_life", parameters.register_new_updatable_parameter(42.0)
        )

    with pytest.raises(ValueError):
        my_updatable.set_internal_parameter(
            "the_meaning_of_life", parameters.register_new_updatable_parameter(41.0)
        )


def test_set_parameter():
    my_updatable = MinimalUpdatable()
    my_updatable.set_parameter(
        "the_meaning_of_life", parameters.register_new_updatable_parameter(42.0)
    )
    with pytest.deprecated_call():
        my_updatable.set_parameter(
            "no_meaning_of_life", parameters.register_new_updatable_parameter()
        )

    assert hasattr(my_updatable, "the_meaning_of_life")
    assert my_updatable.the_meaning_of_life == 42.0

    assert hasattr(my_updatable, "no_meaning_of_life")
    assert my_updatable.no_meaning_of_life is None


def test_update_rejects_internal_parameters():
    my_updatable = MinimalUpdatable()
    my_updatable.set_internal_parameter(
        "the_meaning_of_life", parameters.register_new_updatable_parameter(42.0)
    )
    assert hasattr(my_updatable, "the_meaning_of_life")

    params = ParamsMap({"a": 1.1, "the_meaning_of_life": 34.0})
    with pytest.raises(
        TypeError,
        match="Items of type InternalParameter cannot be modified through update",
    ):
        my_updatable.update(params)

    assert my_updatable.a is None
    assert my_updatable.the_meaning_of_life == 42.0


def test_updatable_collection_is_updated():
    obj: UpdatableCollection = UpdatableCollection([SimpleUpdatable()])
    new_params = {"x": -1.0, "y": 5.5}

    assert not obj.is_updated()
    obj.update(ParamsMap(new_params))
    assert obj.is_updated()


def test_updatablecollection_without_derived_parameters():
    obj: UpdatableCollection = UpdatableCollection()

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

    with pytest.deprecated_call():
        assert base.required_parameters() == RequiredParameters(
            [
                SamplerParameter(name="a"),
                SamplerParameter(name="x"),
                SamplerParameter(name="y"),
                SamplerParameter(name="A"),
                SamplerParameter(name="B"),
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
