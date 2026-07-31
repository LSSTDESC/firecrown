"""Cluster updatables connector."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, TypeGuard, cast

from firecrown.updatable import Updatable, UpdatableCollection
from firecrown.updatable import register_new_updatable_parameter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from firecrown.updatable import InternalParameter, SamplerParameter


class _ClusterParameters(Protocol):
    """Typing contract for the cluster object's ``parameters`` attribute."""

    def __getitem__(self, key: str) -> float: ...  # defaults are floats  # noqa: E704

    def __setitem__(  # noqa: E704
        self,
        key: str,
        value: float | None,
    ) -> None:
        # Registered sampler parameters are None pre-update and float post-update.
        ...


class _ClusterObjectWithParameters(Protocol):
    """Protocol for cluster objects exposing a mutable ``parameters`` attribute."""

    parameters: _ClusterParameters


class _ClusterObjectWithCosmo(_ClusterObjectWithParameters, Protocol):
    """Cluster object protocol for objects exposing a ``cosmo`` attribute."""

    cosmo: object


class _ClusterRecipe(Protocol):
    """Protocol for cluster recipes exposing named cluster objects."""

    def __getattr__(self, name: str) -> object:
        pass


class _ClusterObjectConfig(TypedDict):
    """Configuration describing how to build updatable cluster parameters."""

    recipe_attribute_name: str
    parameters: Iterable[str]
    has_cosmo: NotRequired[bool]


def _is_cluster_object_with_parameters(
    value: object,
) -> TypeGuard[_ClusterObjectWithParameters]:
    """Check whether a value has a ``parameters`` mapping we can update."""
    parameters_attr = getattr(value, "parameters", None)
    if parameters_attr is None:
        return False
    return hasattr(parameters_attr, "__getitem__") and hasattr(
        parameters_attr, "__setitem__"
    )


def _is_cluster_object_with_cosmo(value: object) -> TypeGuard[_ClusterObjectWithCosmo]:
    """Check whether a cluster object supports cosmology injection."""
    return _is_cluster_object_with_parameters(value) and hasattr(value, "cosmo")


def _get_recipe_cluster_object(
    cluster_recipe: _ClusterRecipe,
    recipe_attribute_name: str,
) -> _ClusterObjectWithParameters:
    """Return a typed cluster object from a cluster recipe by attribute name."""
    cluster_object = getattr(cluster_recipe, recipe_attribute_name)
    if not _is_cluster_object_with_parameters(cluster_object):
        raise TypeError(
            f"Cluster recipe attribute '{recipe_attribute_name}' does not expose "
            "a valid 'parameters' mapping."
        )
    return cluster_object


class UpdatableParameters(Updatable):
    """Store and pass updatable parameters to cluster objects.

    :ivar updatable_parameters: Name of updatable parameters.
    :ivar recipe_attribute_name: Attribute name used to look up cluster objects
        from a cluster recipe.
    """

    def __init__(
        self,
        recipe_attribute_name: str,
        updatable_parameters: Iterable[str],
    ) -> None:
        """Create an :class:`UpdatableParameters` helper.

        :param recipe_attribute_name: Name of the attribute in the cluster
            recipe that identifies the cluster object.
        :param updatable_parameters: Names of the updatable parameters to
            register.
        """
        super().__init__()
        self.recipe_attribute_name = recipe_attribute_name
        self.updatable_parameters = list(updatable_parameters)

    def _ini_file_par_name(self, par_name: str) -> str:
        return f"{self.recipe_attribute_name}_{par_name}"

    def init_parameters(self, cluster_object: _ClusterObjectWithParameters) -> None:
        """Instantiate all updatable parameters.

        :param cluster_object: Cluster object providing default values in its
            ``parameters`` mapping.
        """
        for par_name in self.updatable_parameters:
            registered_parameter = cast(
                "SamplerParameter | InternalParameter",
                register_new_updatable_parameter(
                    default_value=cluster_object.parameters[par_name]
                ),
            )
            setattr(
                self,
                self._ini_file_par_name(par_name),
                registered_parameter,
            )

    def export_parameters(self, cluster_object: _ClusterObjectWithParameters) -> None:
        """Pass internal parameters to the cluster object.

        :param cluster_object: Cluster object whose ``parameters`` mapping
            should be updated.
        """
        for par_name in self.updatable_parameters:
            cluster_object.parameters[par_name] = getattr(
                self, self._ini_file_par_name(par_name)
            )


class UpdatableClusterObjects(Updatable):
    """Store updatable parameters for cluster objects.

    This class passes updatable parameters to all objects within a cluster
    recipe.

    :ivar cluster_objects_configs: Sequence of configuration dictionaries
        describing which parameters of each cluster object will be updated.

    Example configuration::

        cluster_objects_configs = (
            {
                "recipe_attribute_name": "mass_distribution",
                "parameters": ["mu0", "mu1", "mu2", "sigma0", "sigma1", "sigma2"],
            },
            {
                "recipe_attribute_name": "cluster_theory",
                "parameters": ["cluster_concentration"],  # if wl profile
                "has_cosmo": True,
            },
            {
                "recipe_attribute_name": "completeness",
                "parameters": ["a_n", "b_n", "a_logm_piv", "b_logm_piv"],
            },
            {
                "recipe_attribute_name": "purity",
                "parameters": ["a_n", "b_n", "a_logm_piv", "b_logm_piv"],
            },
        )
    """

    def __init__(
        self,
        cluster_objects_configs: Sequence[_ClusterObjectConfig],
    ) -> None:
        """Create an :class:`UpdatableClusterObjects` helper.

        :param cluster_objects_configs: Configuration describing which
            attributes on a cluster recipe should be treated as updatable
            cluster objects.
        """
        super().__init__()
        self.cluster_objects_configs = list(cluster_objects_configs)
        self.my_updatables: UpdatableCollection[UpdatableParameters] = (
            UpdatableCollection()
        )
        self._updatables_by_name: dict[str, UpdatableParameters] = {}
        for conf in self.cluster_objects_configs:
            updatable_parameters = UpdatableParameters(
                conf["recipe_attribute_name"],
                conf["parameters"],
            )
            setattr(
                self,
                conf["recipe_attribute_name"],
                updatable_parameters,
            )
            self._updatables_by_name[conf["recipe_attribute_name"]] = (
                updatable_parameters
            )
            self.my_updatables.append(updatable_parameters)

    def init_all_parameters(self, cluster_recipe: _ClusterRecipe) -> None:
        """Instantiate all updatable parameters.

        :param cluster_recipe: Cluster recipe containing the concrete cluster
            objects as attributes.
        """
        for conf in self.cluster_objects_configs:
            cluster_object = _get_recipe_cluster_object(
                cluster_recipe,
                conf["recipe_attribute_name"],
            )
            self._updatables_by_name[conf["recipe_attribute_name"]].init_parameters(
                cluster_object
            )

    def export_all_parameters(
        self,
        cluster_recipe: _ClusterRecipe,
        cosmo: object,
    ) -> None:
        """Export internal parameters to all configured cluster objects.

        :param cluster_recipe: Cluster recipe containing the concrete cluster
            objects as attributes.
        :param cosmo: Cosmology object injected into cluster objects that declare
            ``has_cosmo=True``.
        """
        for conf in self.cluster_objects_configs:
            cluster_object = _get_recipe_cluster_object(
                cluster_recipe,
                conf["recipe_attribute_name"],
            )
            self._updatables_by_name[conf["recipe_attribute_name"]].export_parameters(
                cluster_object
            )
            if conf.get("has_cosmo", False):
                if not _is_cluster_object_with_cosmo(cluster_object):
                    x = conf["recipe_attribute_name"]
                    msg = (
                        f"Cluster recipe attribute '{x}' declares has_cosmo=True "
                        "but does not expose a 'cosmo' attribute."
                    )
                    raise TypeError(msg)
                cluster_object.cosmo = cosmo
