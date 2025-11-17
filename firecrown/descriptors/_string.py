"""String attribute descriptor.

This module provides the TypeString descriptor class for validating string
attributes in classes.
"""

# Validator classes naturally have very few public methods. We silence the
# warning from pylint that complains about this.
#
# pylint: disable=R0903

from collections.abc import Callable


class TypeString:
    """String attribute descriptor.

    :class:`TypeString` provides several different means of validation of the
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
