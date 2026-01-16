"""Utility functions for updatable objects."""

from __future__ import annotations
import warnings

from firecrown.updatable._parameters_map import ParamsMap

from firecrown.updatable._base import Updatable
from firecrown.updatable._collection import UpdatableCollection
from firecrown.updatable._types import UpdatableProtocol


def get_default_params(*args: Updatable) -> dict[str, float]:
    """Get a ParamsMap with the default values of all parameters in the updatables.

    :param args: updatables to get the default parameters from
    :return: a ParamsMap with the default values of all parameters
    """
    updatable_collection = UpdatableCollection(args)
    required_parameters = updatable_collection.required_parameters()
    default_parameters = required_parameters.get_default_values()

    return default_parameters


def get_default_params_map(*args: Updatable) -> ParamsMap:
    """Get a ParamsMap with the default values of all parameters in the updatables.

    :param args: updatables to get the default parameters from
    :return: a ParamsMap with the default values of all parameters
    """
    default_parameters: dict[str, float] = get_default_params(*args)
    return ParamsMap(**default_parameters)


_FINAL_METHODS = (
    "update",
    "reset",
    "is_updated",
    "required_parameters",
    "get_derived_parameters",
)


def assert_updatable_interface(
    obj: UpdatableProtocol,
    recursive: bool = True,
    raise_on_override: bool = False,
):
    """Asserts that all final methods were not overridden.

    The methods:

    - :func:`update`
    - :func:`reset`
    - :func:`is_updated`
    - :func:`required_parameters`
    - :func:`get_derived_parameters`

    Are final and should not be overridden. If any of these methods are
    overridden a TypeError is raised.
    """
    overwritten = []

    my_type: type
    match obj:
        case UpdatableCollection():
            my_type = UpdatableCollection
            if recursive:
                for item in obj:
                    assert_updatable_interface(
                        item, raise_on_override=raise_on_override
                    )
        case Updatable():
            my_type = Updatable
            if recursive:
                # pylint: disable-next=protected-access
                for item in obj._updatables:
                    assert_updatable_interface(
                        item, raise_on_override=raise_on_override
                    )
        case _:
            raise TypeError("Expected Updatable or UpdatableCollection")

    for method in _FINAL_METHODS:
        original_method = getattr(my_type, method)
        instance_attribute = getattr(obj, method, None)
        current_method = getattr(instance_attribute, "__func__", instance_attribute)
        if current_method is not original_method:
            overwritten.append(method)

    msg = f"Updatable interface methods {overwritten} were overridden"

    if overwritten:
        if raise_on_override:
            raise TypeError(msg)
        warnings.warn(msg, RuntimeWarning)
