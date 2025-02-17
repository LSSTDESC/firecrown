"""Type validation as used in connectors.

Validators are created using the constructor for each class.
Access to the data done through the object name, not through any named function.
Setting the data is validated with the class's `validate` function; the user does
not need to call any special functions.

Validators are intended for use in class definitions. An example is a class that
has an attribute `x` that is required to be a float in the range
[1.0, 3.0], but is optional and has a default value of None:

.. code:: python

    class SampleValidatedThing:
        x = TypeFloat(1.0, 3.0, allow_none=True)

        def __init__(self):
            self.x = None

"""

# Validator classes naturally have very few public methods. We silence the
# warning from pylint that complains about this.
#
# pylint: disable=R0903

import math
from typing import Callable


class TypeFloat:
    """Floating point number attribute descriptor."""

    def __init__(
        self,
        minvalue: None | float = None,
        maxvalue: None | float = None,
        allow_none: bool = False,
    ) -> None:
        """Initialize the TypeFloat object.

        :param minvalue: The minimum value allowed for the attribute.
        :param maxvalue: The maximum value allowed for the attribute.
        :param allow_none: Whether the attribute can be None.
        """
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.allow_none = allow_none

    def validate(self, value: None | float) -> None:
        """Run all validators on this value.

        Raise an exception if the provided `value` does not meet all of the
        required conditions enforced by this validator.
        """
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

    def _is_constrained(self) -> bool:
        """Return if this validation enforces any constraint."""
        return not ((self.minvalue is None) and (self.maxvalue is None))

    def __set_name__(self, _, name: str) -> None:
        """Create the name of the private instance variable that will hold the value.

        :param name: The name of the private instance variable to be created.
        """
        self.private_name = "_" + name  # pylint: disable-msg=W0201

    def __get__(self, obj, objtype=None) -> float:
        """Accessor method, which reads controlled value.

        This is invoked whenever the validated variable is read.

        :param obj: The object that contains the validated variable.
        :param objtype: The type of the object that contains the validated variable.
        """
        return getattr(obj, self.private_name)

    def __set__(self, obj, value: None | float) -> None:
        """Setter for the validated variable.

        This function invokes the `validate` method of the derived class.

        :param obj: The object that contains the validated variable.
        :param value: The new value of the validated variable.
        """
        self.validate(value)
        setattr(obj, self.private_name, value)


class TypeString:
    """String attribute descriptor.

    :classs:`TypeString` provides several different means of validation of the
    controlled string attribute, all of which are optional.
    """

    def __init__(
        self,
        minsize: None | int = None,
        maxsize: None | int = None,
        predicate: None | Callable[[str], bool] = None,
    ) -> None:
        """Initialize the TypeString object.

        :param minsize: The minimum length allowed for the string.
        :param maxsize: The maximum length allowed for the string.
        :param predicate: A function that returns true if the string is allowed.
        """
        self.minsize = minsize
        self.maxsize = maxsize
        self.predicate = predicate

    def validate(self, value: None | str) -> None:
        """Run all validators on this value.

        Raise an exception if the provided `value` does not meet all of the
        required conditions enforced by this validator.

        :param value: The value to be validated.
        """
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

    def __set_name__(self, _, name: str) -> None:
        """Create the name of the private instance variable that will hold the value.

        :param name: The name of the private instance variable to be created.
        """
        self.private_name = "_" + name  # pylint: disable-msg=W0201

    def __get__(self, obj, objtype=None) -> str:
        """Accessor method, which reads controlled value.

        This is invoked whenever the validated variable is read.

        :param obj: The object that contains the validated variable.
        :param objtype: The type of the object that contains the validated variable.
        """
        return getattr(obj, self.private_name)

    def __set__(self, obj, value: None | str) -> None:
        """Setter for the validated variable.

        This function invokes the `validate` method of the derived class.

        :param obj: The object that contains the validated variable.
        :param value: The new value of the validated variable.
        """
        self.validate(value)
        setattr(obj, self.private_name, value)
