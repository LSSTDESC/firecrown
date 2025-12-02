"""Parameter types and creation functions."""

from ._names import parameter_get_full_name


class SamplerParameter:
    """Class to represent a sampler defined parameter."""

    def __init__(
        self,
        *,
        default_value: float,
        name: None | str = None,
        prefix: None | str = None,
        shared: bool = True,
    ):
        """Creates a new SamplerParameter instance.

        This represents a parameter having its value defined by the sampler.

        :param default_value: the default value of the parameter
        :param name: the name of the parameter
        :param prefix: the prefix of the parameter
        :param shared: if False, the parameter will not receive a prefix,
            making it the same across all instances
        """
        self._prefix = prefix
        self._name = name
        self.default_value = default_value
        self._shared = shared

    def get_default_value(self) -> float:
        """Get the default value of this parameter."""
        return self.default_value

    def set_fullname(self, prefix: str | None, name: str):
        """Set the prefix and name of this parameter.

        If the parameter is not shared (shared=False), the prefix will be
        ignored and the parameter will have the same name across all instances.

        :param prefix: new prefix (ignored if shared=False)
        :param name: the name of the parameter
        """
        if self._shared:
            self._prefix = prefix
        else:
            self._prefix = None
        self._name = name

    @property
    def shared(self) -> bool:
        """Get whether this parameter is shared across instances."""
        return self._shared

    @property
    def prefix(self) -> str | None:
        """Get the prefix of this parameter."""
        return self._prefix

    @property
    def name(self) -> str:
        """Get the name of this parameter."""
        if self._name is None:
            raise ValueError("Parameter name is not set")
        return self._name

    @property
    def fullname(self) -> str:
        """Get the full name of this parameter."""
        return parameter_get_full_name(self.prefix, self.name)

    def __hash__(self) -> int:
        """Return the hash of the full name of this parameter."""
        return hash(self.fullname)

    def __eq__(self, other: object) -> bool:
        """Return whether this parameter is equal to another.

        Two SamplerParameter objects are equal if they have the same full name.
        """
        if not isinstance(other, SamplerParameter):
            raise NotImplementedError(
                f"SamplerParameter comparison is only implemented for "
                f"SamplerParameter objects, received {type(other)}"
            )
        return (
            self.fullname == other.fullname
            and self.default_value == other.default_value
            and self._prefix == other._prefix
            and self._name == other._name
            and self._shared == other._shared
        )


class InternalParameter:
    """Class to represent an internally defined parameter."""

    def __init__(self, value: float):
        """Creates a new InternalParameter instance.

        This represents an internal parameter with its value defined by value.
        """
        self.value = value

    def set_value(self, value: float):
        """Set the value of this parameter.

        :param value: new value
        """
        self.value = value

    def get_value(self) -> float:
        """Return the current value of this parameter."""
        return self.value


def register_new_updatable_parameter(
    value: None | float = None, *, default_value: float, shared: bool = True
):
    """Create a new parameter, either a SamplerParameter or an InternalParameter.

    If `value` is `None`, the result will be a `SamplerParameter`; Firecrown
    will expect this value to be supplied by the sampling framework. If `value`
    is a `float` quantity, then Firecrown will expect this parameter to *not*
    be supplied by the sampling framework, and instead the provided value will
    be used for every sample.

    Only `None` or a `float` value is allowed.

    :param value: the value of the parameter
    :param default_value: the default value of the parameter to be used
        if `value` is `None`
    :param shared: if False, the parameter will not receive a prefix,
        making it the same across all instances (only applies when value is None)
    :return: a `SamplerParameter` if `value` is `None`, otherwise an `InternalParameter`
    :raises TypeError: if `value` is not `None` and not a `float`
    """
    result: SamplerParameter | InternalParameter
    if value is None:
        result = SamplerParameter(default_value=default_value, shared=shared)

    elif not isinstance(value, float):
        raise TypeError(
            f"parameter.create() requires a float parameter or none, "
            f"not {type(value)}"
        )
    else:
        result = InternalParameter(value)
    assert result is not None
    return result
