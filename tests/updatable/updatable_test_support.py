"""Shared concrete Updatable test classes."""

from typing import cast

from firecrown import updatable
from firecrown.updatable import DerivedParameter, DerivedParameterCollection, Updatable


class MinimalUpdatable(Updatable):
    """A concrete time that implements Updatable."""

    def __init__(self, prefix: str | None = None):
        """Initialize object with defaulted value."""
        super().__init__(prefix)
        self.a = updatable.register_new_updatable_parameter(default_value=1.0)


class SimpleUpdatable(Updatable):  # pylint: disable=too-many-instance-attributes
    """A concrete type that implements Updatable."""

    def __init__(self, prefix: str | None = None):
        """Initialize object with defaulted values."""
        super().__init__(prefix)

        self.x = updatable.register_new_updatable_parameter(default_value=2.0)
        self.y = updatable.register_new_updatable_parameter(default_value=3.0)


class UpdatableWithDerived(Updatable):
    """A concrete type that implements Updatable that implements derived parameters."""

    def __init__(self):
        """Initialize object with defaulted values."""
        super().__init__()

        # These attributes hold SamplerParameter until update() runs, after
        # which Updatable.__setattr__ replaces them with their float value.
        self.A: float = cast(
            float, updatable.register_new_updatable_parameter(default_value=2.0)
        )
        self.B: float = cast(
            float, updatable.register_new_updatable_parameter(default_value=1.0)
        )

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_scale = DerivedParameter("Section", "Name", self.A + self.B)
        derived_parameters = DerivedParameterCollection([derived_scale])

        return derived_parameters
