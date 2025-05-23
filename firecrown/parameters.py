"""Classes and functions to support groups of named parameters.

These are used in Firecrown in preference to the Python dictionary in order to
provide better type safety.
"""

from __future__ import annotations
from typing import Iterable, Iterator, Sequence


def parameter_get_full_name(prefix: None | str, param: str) -> str:
    """Form a full parameter name from the given (optional) prefix and name.

    Parameter names, as stored in SACC, for example, contain an optional
    prefix; if a prefix is present, it will be separated from the name by
    an underscore.

    Prefixes and names should avoid containing embedded underscores. This
    is currently not enforced in the code.

    The parameter name can not be empty, even if accompanied by a prefix;
    this is enforced in the code.

    :param prefix: optional prefix
    :param param: name
    :return: full name
    :raises ValueError: if the parameter name is empty
    """
    if len(param) == 0:
        raise ValueError("param must not be an empty string")

    if prefix:
        return f"{prefix}_{param}"
    return param


def _validate_params_map_value(name: str, value: float | list[float]) -> None:
    """Check if the value is a float or a list of floats.

    Raises a TypeError if the value is not a float or a list of floats.

    :param name: name of the parameter
    :param value: value to be checked
    """
    if not isinstance(value, (float, list)):
        raise TypeError(
            f"Value for parameter {name} is not a float or a list of floats: "
            f"{type(value)}"
        )

    if isinstance(value, list):
        if not all(isinstance(v, float) for v in value):
            raise TypeError(
                f"Value for parameter {name} is not a float or a list of floats: "
                f"{type(value)}"
            )


class ParamsMap(dict[str, float]):
    """A specialized dict in which all keys are strings and values are floats.

    The recommended access method is get_from_prefix_param, rather than indexing
    with square brackets like x[].
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for name, value in self.items():
            _validate_params_map_value(name, value)

        self.lower_case: bool = False

    def use_lower_case_keys(self, enable: bool) -> None:
        """Control whether keys will be translated into lower case.

        If `enable` is True, such translation will be done.
        This can help make sure code works with CosmoSIS, because such translation
        is done inside CosmoSIS itself.

        :param enable: whether to enable or disable this option
        """
        self.lower_case = enable

    def get_from_full_name(self, full_name: str) -> float:
        """Return the parameter identified by the full name.

        Raises a KeyError if the parameter is not found.
        """
        if full_name in self.keys():
            return self[full_name]

        if self.lower_case:
            full_name_lower = full_name.lower()
            if full_name_lower in self.keys():
                return self[full_name_lower]

        raise KeyError(f"Key {full_name} not found.")

    def get_from_prefix_param(self, prefix: None | str, param: str) -> float:
        """Return the parameter identified by the optional prefix and parameter name.

        See parameter_get_full_name for rules on the forming of prefix and name.
        Raises a KeyError if the parameter is not found.
        """
        fullname = parameter_get_full_name(prefix, param)
        return self.get_from_full_name(fullname)


class RequiredParameters:
    """Represents a sequence of parameter names.

    This class provides some type safety to distinguish between an arbitrary
    list of strings, and one intended to be a list of required parameter names.

    An instance can be created from a list of strings.
    Instances can be concatenated using `+`, and compared for equality using `==`.

    To iterate through the names (which are strings), use `get+params_names`,
    which implements lazy evaluation.
    """

    def __init__(self, params: Iterable[SamplerParameter]):
        """Construct an instance from an Iterable yielding strings."""
        self.params_set: set[SamplerParameter] = set(params)

    def __len__(self):
        """Return the number of parameters contained."""
        return len(self.params_set)

    def __add__(self, other: RequiredParameters) -> RequiredParameters:
        """Return a new RequiredParameters with the concatenated names.

        Note that this function returns a new object that does not share state
        with either argument to the addition operator.
        """
        return RequiredParameters(self.params_set | other.params_set)

    def __sub__(self, other: RequiredParameters) -> RequiredParameters:
        """Return a new RequiredParameters with the names in self but not in other.

        Note that this function returns a new object that does not share state
        with either argument to the subtraction operator.
        """
        return RequiredParameters(self.params_set - other.params_set)

    def __eq__(self, other: object):
        """Compare two RequiredParameters objects for equality.

        This implementation raises a NotImplemented exception unless both
        objects are RequireParameters objects.

        Two RequireParameters objects are equal if their contained names
        are equal (including appearing in the same order).
        """
        if not isinstance(other, RequiredParameters):
            n = type(other).__name__
            raise TypeError(
                f"Cannot compare a RequiredParameter to an object of type {n}"
            )
        return self.params_set == other.params_set

    def get_params_names(self) -> Iterator[str]:
        """Implement lazy iteration through the contained parameter names."""
        params_names_set = set(parameter.fullname for parameter in self.params_set)
        yield from params_names_set

    def get_default_values(self) -> dict[str, float]:
        """Return a dictionary with the default values of the parameters."""
        default_values = {}
        for parameter in self.params_set:
            default_values[parameter.fullname] = parameter.get_default_value()

        return default_values


class DerivedParameter:
    """Represents a derived scalar parameter generated by an Updatable object.

    This class provide the type that encapsulate a derived scalar quantity (represented
     by a float) computed by an Updatable object during a statistical analysis.
    """

    def __init__(self, section: str, name: str, val: float):
        """Initialize the DerivedParameter object."""
        self.section: str = section
        self.name: str = name
        if not isinstance(val, float):
            raise TypeError(
                "DerivedParameter expects a float but received a " + str(type(val))
            )
        self.val: float = val

    def get_val(self) -> float:
        """Return the value of this parameter."""
        return self.val

    def __eq__(self, other: object) -> bool:
        """Compare two DerivedParameter objects for equality.

        This implementation raises a NotImplemented exception unless both
        objects are DerivedParameter objects.

        Two DerivedParameter objects are equal if they have the same
        section, name and value.
        """
        if not isinstance(other, DerivedParameter):
            raise NotImplementedError(
                "DerivedParameter comparison is only implemented for "
                "DerivedParameter objects"
            )
        return (
            self.section == other.section
            and self.name == other.name
            and self.val == other.val
        )

    def get_full_name(self):
        """Constructs the full name using section--name."""
        return f"{self.section}--{self.name}"


class DerivedParameterCollection:
    """Represents a list of DerivedParameter objects."""

    def __init__(self, derived_parameters: Sequence[DerivedParameter]):
        """Construct an instance from a sequence of DerivedParameter objects."""
        if not all(isinstance(x, DerivedParameter) for x in derived_parameters):
            raise TypeError(
                "DerivedParameterCollection expects a list of DerivedParameter"
                "but received a " + str([str(type(x)) for x in derived_parameters])
            )

        self.derived_parameters: dict[str, DerivedParameter] = {}

        for derived_parameter in derived_parameters:
            self.add_required_parameter(derived_parameter)

    def __len__(self) -> int:
        """Return the number of parameters contained."""
        return len(self.derived_parameters)

    def __add__(self, other: None | DerivedParameterCollection):
        """Add two DerivedParameterCollection objects.

        Return a new DerivedParameterCollection with the lists of DerivedParameter
        objects.

        If other is none return self. Otherwise, constructs a new object representing
        the addition.

        Note that this function returns a new object that does not share state
        with either argument to the addition operator.
        """
        if other is None:
            return self

        return DerivedParameterCollection(
            list(self.derived_parameters.values())
            + list(other.derived_parameters.values())
        )

    def __eq__(self, other: object):
        """Compare two DerivedParameterCollection objects for equality.

        This implementation raises a NotImplemented exception unless both
        objects are DerivedParameterCollection objects.

        Two DerivedParameterCollection objects are equal if they contain the same
        DerivedParameter objects.
        """
        if not isinstance(other, DerivedParameterCollection):
            raise NotImplementedError(
                "DerivedParameterCollection comparison is only implemented for "
                "DerivedParameterCollection objects"
            )
        return self.derived_parameters == other.derived_parameters

    def __iter__(self) -> Iterator[tuple[str, str, float]]:
        """Implementation of lazy iteration through the collection."""
        for derived_parameter in self.derived_parameters.values():
            yield (
                derived_parameter.section,
                derived_parameter.name,
                derived_parameter.get_val(),
            )

    def add_required_parameter(self, derived_parameter: DerivedParameter):
        """Adds derived_parameter to the collection.

        We raises an ValueError if a required parameter with the same name is already
        present in the collection.
        """
        required_parameter_full_name = derived_parameter.get_full_name()
        if required_parameter_full_name in self.derived_parameters:
            raise ValueError(
                f"RequiredParameter named {required_parameter_full_name}"
                f" is already present in the collection"
            )
        self.derived_parameters[required_parameter_full_name] = derived_parameter

    def get_derived_list(self) -> list[DerivedParameter]:
        """Implement lazy iteration through the contained parameter names."""
        return list(self.derived_parameters.values())


class SamplerParameter:
    """Class to represent a sampler defined parameter."""

    def __init__(
        self,
        *,
        default_value: float,
        name: None | str = None,
        prefix: None | str = None,
    ):
        """Creates a new SamplerParameter instance.

        This represents a parameter having its value defined by the sampler.
        """
        self._prefix = prefix
        self._name = name
        self.default_value = default_value

    def get_default_value(self) -> float:
        """Get the default value of this parameter."""
        return self.default_value

    def set_fullname(self, prefix: str | None, name: str):
        """Set the prefix of this parameter.

        :param prefix: new prefix
        """
        self._prefix = prefix
        self._name = name

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
    value: None | float = None, *, default_value: float
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
    :return: a `SamplerParameter` if `value` is `None`, otherwise an `InternalParameter`
    :raises TypeError: if `value` is not `None` and not a `float`
    """
    result: SamplerParameter | InternalParameter
    if value is None:
        result = SamplerParameter(default_value=default_value)

    elif not isinstance(value, float):
        raise TypeError(
            f"parameter.create() requires a float parameter or none, "
            f"not {type(value)}"
        )
    else:
        result = InternalParameter(value)
    assert result is not None
    return result
