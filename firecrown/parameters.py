"""Classes and functions to support groups of named parameters.

"""

from __future__ import annotations
from typing import Iterable, List, Dict, Set, Tuple, Optional, Iterator
from abc import ABC, abstractmethod


def parameter_get_full_name(prefix: Optional[str], param: str) -> str:
    """Form a full parameter name from the given (optional) prefix and name.

    Parameter names, as stored in SACC, for example, contain an optional
    prefix; if a prefix is present, it will be separated from the name by
    an underscore.

    Prefixes and names should avoid containing embedded underscores. This
    is currently not enforced in the code.

    The parameter name can not be empty, even if accompanied by a prefix;
    this is enforced in the code.

    Ill-formed parameter names result in raising a ValueError.
    """
    if len(param) == 0:
        raise ValueError("param must not be an empty string")

    if prefix:
        return f"{prefix}_{param}"
    return param


class ParamsMap(Dict[str, float]):
    """A specialized Dict in which all keys are strings and values are floats.

    The recommended access method is get_from_prefix_param, rather than indexing
    with square brackets like x[].
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lower_case: bool = False

    def use_lower_case_keys(self, enable: bool) -> None:
        """Control whether keys will be translated into lower case.
        If `enable` is True, such translation will be done.
        This can help make sure code works with CosmoSIS, because such translation
        is done inside CosmoSIS itself."""
        self.lower_case = enable

    def get_from_prefix_param(self, prefix: Optional[str], param: str) -> float:
        """Return the parameter identified by the optional prefix and parameter name.


        See parameter_get_full_name for rules on the forming of prefix and name.
        Raises a KeyError if the parameter is not found.
        """
        fullname = parameter_get_full_name(prefix, param)
        if self.lower_case:
            fullname = fullname.lower()

        if fullname in self.keys():
            return self[fullname]
        raise KeyError(
            f"Prefix `{prefix}`, param `{param}', key `{fullname}' not found."
        )


class RequiredParameters:
    """Represents a sequence of parameter names.

    This class provides some type safety to distinguish between an arbitrary
    list of strings, and one intended to be a list of required parameter names.

    An instance can be created from a list of strings.
    Instances can be concatenated using `+`, and compared for equality using `==`.

    To iterate through the names (which are strings), use `get+params_names`,
    which implements lazy evaluation.
    """

    def __init__(self, params_names: Iterable[str]):
        """Construct an instance from an Iterable yielding strings."""
        self.params_names: Set[str] = set(params_names)

    def __add__(self, other: RequiredParameters):
        """Return a new RequiredParameters with the concatenated names.

        Note that this function returns a new object that does not share state
        with either argument to the addition operator."""
        return RequiredParameters(self.params_names | other.params_names)

    def __eq__(self, other: object):
        """Compare two RequiredParameters objects for equality.

        This implementation raises a NotImplemented exception unless both
        objects are RequireParameters objects.

        Two RequireParameters objects are equal if their contained names
        are equal (including appearing in the same order).
        """
        if not isinstance(other, RequiredParameters):
            return NotImplemented
        return self.params_names == other.params_names

    def get_params_names(self):
        """Implement lazy iteration through the contained parameter names."""
        params_names_set = set(self.params_names)

        for name in params_names_set:
            yield name


class DerivedParameter(ABC):
    """Represents a derived parameter generated by an Updatable object

    This class provide the type that encapsulate a derived quantity computed
    by an Updatable object during a statistical analysis.
    """

    def __init__(self, section: str, name: str):
        """Constructs a new derived parameter."""
        self.section: str = section
        self.name: str = name

    def get_full_name(self):
        """Constructs the full name using section--name."""
        return f"{self.section}--{self.name}"

    @abstractmethod
    def get_val(self):
        """Returns the value contained."""


class DerivedParameterScalar(DerivedParameter):
    """Represents a derived scalar parameter generated by an Updatable object

    This class provide the type that encapsulate a derived scalar quantity (represented
     by a float) computed by an Updatable object during a statistical analysis.
    """

    def __init__(self, section: str, name: str, val: float):
        super().__init__(section, name)

        if not isinstance(val, float):
            raise TypeError(
                "DerivedParameterScalar expects a float but received a "
                + str(type(val))
            )
        self.val: float = val

    def get_val(self) -> float:
        return self.val


class DerivedParameterCollection:
    """Represents a list of DerivedParameter objects."""

    def __init__(self, derived_parameters: List[DerivedParameter]):
        """Construct an instance from a List of DerivedParameter objects."""

        if not all(isinstance(x, DerivedParameter) for x in derived_parameters):
            raise TypeError(
                "DerivedParameterCollection expects a list of DerivedParameter but "
                "received a " + str([str(type(x)) for x in derived_parameters])
            )

        self.derived_parameters: Dict[str, DerivedParameter] = {}

        for parameter in derived_parameters:
            self.add_required_parameter(parameter)

    def __add__(self, other: Optional[DerivedParameterCollection]):
        """Return a new DerivedParameterCollection with the lists of DerivedParameter
        objects.

        If other is none return self. Otherwise, constructs a new object representing
        the addition.

        Note that this function returns a new object that does not share state
        with either argument to the addition operator."""
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
            return NotImplemented
        return self.derived_parameters == other.derived_parameters

    def __iter__(self) -> Iterator[Tuple[str, str, float]]:
        for derived_parameter in self.derived_parameters.values():
            yield (
                derived_parameter.section,
                derived_parameter.name,
                derived_parameter.get_val(),
            )

    def add_required_parameter(self, derived_parameter: DerivedParameter):
        """Adds derived_parameter to the collection, it raises an ValueError if a
        required parameter with the same name is already present in the collection.
        """

        required_parameter_full_name = derived_parameter.get_full_name()
        if required_parameter_full_name in self.derived_parameters:
            raise ValueError(
                f"RequiredParameter named {required_parameter_full_name}"
                f" is already present in the collection"
            )
        self.derived_parameters[required_parameter_full_name] = derived_parameter

    def get_derived_list(self) -> List[DerivedParameter]:
        """Implement lazy iteration through the contained parameter names."""

        return list(self.derived_parameters.values())


class SamplerParameter:
    """Class to represent a sampler defined parameter."""

    def __init__(self):
        """Creates a new SamplerParameter instance that represents a parameter
        having its value defined by the sampler."""
        self.value = None

    def set_value(self, value: float):
        """Set the value of this parameter."""
        self.value = value

    def get_value(self) -> float:
        """Get the current value of this parameter."""
        return self.value


class InternalParameter:
    """Class to represent an internally defined parameter."""

    def __init__(self, value: float):
        """Creates a new InternalParameter instance that represents an
        internal parameter with its value defined by value."""
        self.value = value

    def set_value(self, value: float):
        """Set the value of this parameter."""
        self.value = value

    def get_value(self) -> float:
        """Return the current value of this parameter."""
        return self.value


def create(value: Optional[float] = None):
    """Create a new parameter.

    If `value` is `None`, the result will be a `SamplerParameter`; Firecrown
    will expect this value to be supplied by the sampling framwork. If `value`
    is a `float` quantity, then Firecrown will expect this parameter to *not*
    be supplied by the sampling framework, and instead the provided value will
    be used for every sample.

    Only `None` or a `float` value is allowed.
    """
    if value is None:
        return SamplerParameter()
    return InternalParameter(value)
