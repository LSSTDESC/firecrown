"""


Descriptors Module
==================

Provides type validation as used in connectors.

Classes provided:
    Validator: Abstract base class for concrete validators
    TypeFloat: verifies a parameter is a float, with optional range checking
    TypeString: verifies a parameter is a string, with optional value checking
    TypeLikelihood: verifies a parameter is a Firecrown likelihood object

Validators are created using the constructor for each class.
Access to the data done through the object name, not through any named function.
Setting the data is validated with the class's `validate` function; the user does
not need to call any special functions.

Validators are intended for use in class definitions. An example is a class that
has a data member `x` that is required to be a float in the range [1.0, 3.0], but
is optional and has a default value of None:

    class SampleValidatedTHing:
        x = TypeFloat(1.0, 3.0, allow_none=True)

        def __init__(self):
            self.x = None

"""

import math
from abc import ABC, abstractmethod
from .likelihood.likelihood import Likelihood


class Validator(ABC):
    """Abstract base class for all validators.

    The base class implements the methods required for setting values, which should
    not be overridden in derived classes.

    Derived classes must implement the abstract method validate, which is called by
    the base class when any variable protected by a Validator is set.
    """

    def __set_name__(self, owner, name):
        """Create the name of the private data member that will hold the value."""
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        """Accessor method, which reads controlled value.

        This is invoked whenever the validated variable is read."""
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        """Setter for the validated variable.

        This function invokes the `validate` method of the derived class."""
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        """Abstract method to perform whatever validation is required."""


class TypeFloat(Validator):
    """Floating point number attribute descriptor."""

    def __init__(self, minvalue=None, maxvalue=None, allow_none=False):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.allow_none = allow_none

    def validate(self, value):
        if self.allow_none and value is None:
            return
        if not isinstance(value, float):
            raise TypeError(f"Expected {value!r} to be a float")
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(f"Expected {value!r} to be at least {self.minvalue!r}")
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(f"Expected {value!r} to be no more than {self.maxvalue!r}")
        if self._is_constrained() and math.isnan(value):
            raise ValueError("NaN is disallowed in a constrained float")

    def _is_constrained(self):
        return not ((self.minvalue is None) and (self.maxvalue is None))


class TypeString(Validator):
    """String attribute descriptor."""

    def __init__(self, minsize=None, maxsize=None, predicate=None):
        self.minsize = minsize
        self.maxsize = maxsize
        self.predicate = predicate

    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be an str")
        if self.minsize is not None and len(value) < self.minsize:
            raise ValueError(
                f"Expected {value!r} to be no smaller than {self.minsize!r}"
            )
        if self.maxsize is not None and len(value) > self.maxsize:
            raise ValueError(
                f"Expected {value!r} to be no bigger than {self.maxsize!r}"
            )
        if self.predicate is not None and not self.predicate(value):
            raise ValueError(f"Expected {self.predicate} to be true for {value!r}")


class TypeLikelihood(Validator):
    """Likelihood attribute descriptor."""

    def __init__(self):
        pass

    def validate(self, value):
        if not isinstance(value, Likelihood):
            raise TypeError(
                f"Expected {value!r} {value} {self} to be a "
                f"firecrown.likelihood.likelihood.Likelihood"
            )
