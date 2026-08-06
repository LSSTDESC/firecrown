"""Tests for firecrown.likelihood._updatable_wrapper."""

import pytest

from firecrown.likelihood._updatable_wrapper import (
    UpdatableClusterObjects,
    _ClusterObjectConfig,
    _is_cluster_object_with_parameters,
)


class _StubParameters:
    """Minimal parameters mapping backed by a plain dict."""

    def __init__(self, data: dict[str, float]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __setitem__(self, key: str, value: float | None) -> None:
        self._data[key] = value  # type: ignore[assignment]

    def keys(self) -> list[str]:
        """Return the parameter names."""
        return list(self._data.keys())


class _StubClusterObject:
    """Cluster object stub with a valid ``parameters`` mapping."""

    def __init__(self, params: dict[str, float]) -> None:
        self.parameters = _StubParameters(params)


class _StubClusterObjectWithCosmo(_StubClusterObject):
    """Cluster object stub that also exposes a ``cosmo`` attribute."""

    def __init__(self, params: dict[str, float]) -> None:
        super().__init__(params)
        self.cosmo: object = None


class _StubRecipe:
    """Cluster recipe stub whose named attributes are set dynamically."""

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


def _make_recipe(**kwargs: object) -> _StubRecipe:
    """Return a :class:`_StubRecipe` with the given attributes pre-set."""
    recipe = _StubRecipe()
    for name, value in kwargs.items():
        object.__setattr__(recipe, name, value)
    return recipe


def test_no_parameters_attr_returns_false() -> None:
    """_is_cluster_object_with_parameters returns False for objects without parameters.

    An object that has no ``parameters`` attribute at all must cause the
    function to return ``False`` rather than raising an exception.
    """
    result = _is_cluster_object_with_parameters(object())
    assert result is False


def test_invalid_recipe_attribute_raises() -> None:
    """init_all_parameters raises TypeError when a recipe attribute lacks parameters.

    If the named attribute on the cluster recipe exists but does not expose a
    valid ``parameters`` mapping, :class:`UpdatableClusterObjects` must raise
    a :class:`TypeError` with a message that identifies the offending attribute.
    """
    configs: list[_ClusterObjectConfig] = [
        {"recipe_attribute_name": "mass_distribution", "parameters": ["mu0"]},
    ]
    cluster_object = _StubClusterObject({"mu0": 1.0})
    uco = UpdatableClusterObjects(configs)
    uco.init_all_parameters(_make_recipe(mass_distribution=cluster_object))

    recipe_with_bad_attr = _make_recipe(mass_distribution=object())
    with pytest.raises(
        TypeError,
        match="Cluster recipe attribute 'mass_distribution' does not expose",
    ):
        uco.init_all_parameters(recipe_with_bad_attr)


def test_has_cosmo_without_cosmo_attr_raises() -> None:
    """export_all_parameters raises TypeError when has_cosmo=True but cosmo is absent.

    When a config entry declares ``has_cosmo=True`` and the corresponding
    cluster object has a valid ``parameters`` mapping but no ``cosmo``
    attribute, :meth:`UpdatableClusterObjects.export_all_parameters` must
    raise a :class:`TypeError` with a message that identifies the offending
    attribute.
    """
    configs: list[_ClusterObjectConfig] = [
        {
            "recipe_attribute_name": "cluster_theory",
            "parameters": ["alpha"],
            "has_cosmo": True,
        },
    ]
    cluster_object = _StubClusterObject({"alpha": 0.5})
    uco = UpdatableClusterObjects(configs)
    uco.init_all_parameters(_make_recipe(cluster_theory=cluster_object))

    params = uco.my_updatables[0]
    from firecrown.updatable import get_default_params_map

    uco.update(get_default_params_map(params))

    with pytest.raises(
        TypeError,
        match="Cluster recipe attribute 'cluster_theory' declares has_cosmo=True",
    ):
        uco.export_all_parameters(
            _make_recipe(cluster_theory=cluster_object),
            cosmo=object(),
        )
