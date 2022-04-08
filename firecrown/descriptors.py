from abc import ABC, abstractmethod
from .likelihood.likelihood import Likelihood


class Validator(ABC):
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class TypeFloat(Validator):
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


class TypeString(Validator):
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
    def __init__(self):
        pass

    def validate(self, value):
        if not isinstance(value, Likelihood):
            raise TypeError(
                f"Expected {value!r} {value} {self} to be a "
                f"firecrown.likelihood.likelihood.Likelihood"
            )
