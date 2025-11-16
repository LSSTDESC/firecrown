"""Floating point number attribute descriptor.

This module provides the TypeFloat descriptor class for validating floating
point attributes in classes.
"""

# Validator classes naturally have very few public methods. We silence the
# warning from pylint that complains about this.
#
# pylint: disable=R0903

from firecrown.descriptors._base import math


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
