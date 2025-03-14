"""
Tests for the module firecrown.descriptors.
"""

import math
import pytest
from firecrown.descriptors import TypeFloat, TypeString


class NotALikelihood:
    """A trivial class that does not inherit from Likelihood."""


class HasUOF:
    """Test harness for unconstrained optional float"""

    x = TypeFloat(allow_none=True)

    def __init__(self):
        """Initialize a HasUOF object with default value `self.x=None`."""
        self.x = None


class HasCOF:
    """Test harness for constrained optional float"""

    x = TypeFloat(1.0, 3.0, allow_none=True)

    def __init__(self):
        """Initialize a HasCOF object with default value of `self.x=None`."""
        self.x = None


class HasURF:
    """Test harness for unconstrained required float"""

    x = TypeFloat()

    def __init__(self):
        """Initialize a HasURF object with default value `self.x=0.0`."""
        self.x = 0.0


class HasCRF:
    """Test harness for constrained required float"""

    x = TypeFloat(1.0, 3.0)

    def __init__(self):
        """Initialize a HasCRF with the default value `self.x=2.0`."""
        self.x = 2.0


class HasLowerBound:
    """Test harness for float with only a lower bound"""

    x = TypeFloat(minvalue=0.0)

    def __init__(self):
        """Initialize a HasLowerBound object with the default value
        `self.x=0.0`."""
        self.x = 0.0


class HasUpperBound:
    """Test harness for float with only an upper bound"""

    x = TypeFloat(maxvalue=1.0)

    def __init__(self):
        """Initialize a HasUpperBound object with the default value
        `self.x=1.0`."""
        self.x = 1.0


class NotStringy:
    """A class that can not be turned into a string."""

    def __str__(self):
        """Does not return a string"""


class HasString:
    """Test harness for string descriptors."""

    x = TypeString()


class HasShortBound:
    """Test harness for minimum length string."""

    x = TypeString(minsize=4)


class HasLongBound:
    """Test harness for maximum length string."""

    x = TypeString(maxsize=2)


class HasLongAndShortBounds:
    """Test harness for both minimum and maximum length string."""

    x = TypeString(minsize=3, maxsize=5)


class StartsWithCow:
    """Test harness that only accepts strings that start with 'cow'."""

    x = TypeString(predicate=lambda s: s.startswith("cow"))


def test_unconstrained_optional_float():
    d = HasUOF()
    with pytest.raises(TypeError):
        d.x = 3  # not a literal float
    assert d.x is None

    d.x = 3.0
    assert d.x == 3.0

    d.x = math.nan
    assert math.isnan(d.x)

    d.x = math.inf
    assert d.x == math.inf


def test_constrained_optional_float():
    d = HasCOF()
    with pytest.raises(TypeError):
        d.x = 3  # not a literal float
    assert d.x is None

    d.x = 3.0
    assert d.x == 3.0
    with pytest.raises(ValueError):
        d.x = -1.0
    assert d.x == 3.0

    with pytest.raises(ValueError):
        d.x = math.nan
    assert d.x == 3.0

    with pytest.raises(ValueError):
        d.x = math.inf  # not in range
    assert d.x == 3.0


def test_unconstrained_required_float():
    d = HasURF()
    with pytest.raises(TypeError):
        d.x = 3  # not a literal float
    assert d.x == 0.0

    d.x = 1.0
    assert d.x == 1.0
    d.x = math.nan
    assert math.isnan(d.x)
    d.x = math.inf
    assert d.x == math.inf


def test_constrained_required_float():
    d = HasCRF()
    with pytest.raises(TypeError):
        d.x = 3  # not a literal float
    assert d.x == 2.0

    d.x = 3.0
    assert d.x == 3.0
    with pytest.raises(ValueError):
        d.x = -10.0
    assert d.x == 3.0

    with pytest.raises(ValueError):
        d.x = math.nan
    assert d.x == 3.0

    with pytest.raises(ValueError):
        d.x = math.inf  # not in range
    assert d.x == 3.0


def test_lower_bound_float():
    d = HasLowerBound()
    assert d.x == 0.0

    d.x = 100.0
    assert d.x == 100.0

    with pytest.raises(ValueError):
        d.x = -1.0
    assert d.x == 100.0

    d.x = math.inf
    assert d.x == math.inf

    with pytest.raises(ValueError):
        d.x = math.nan
    assert d.x == math.inf


def test_upper_bound_float():
    d = HasUpperBound()
    assert d.x == 1.0

    d.x = -1.0
    assert d.x == -1.0

    with pytest.raises(ValueError):
        d.x = 100.0
    assert d.x == -1.0

    d.x = -math.inf
    assert d.x == -math.inf


def test_reading_string():
    d = HasString()
    d.x = "cow"

    assert d.x == "cow"


def test_string_conversion_failure():
    d = HasString()
    with pytest.raises(TypeError):
        # We ignore type checking on this line because we are testing the error
        # handling for the very type error that mypy would detect.
        d.x = NotStringy()  # type: ignore


def test_string_too_short():
    d = HasShortBound()
    d.x = "walrus"
    with pytest.raises(ValueError):
        d.x = "cow"


def test_string_too_long():
    d = HasLongBound()
    d.x = "ou"
    with pytest.raises(ValueError):
        d.x = "cow"


def test_string_predicate():
    d = StartsWithCow()
    d.x = "cowabunga"
    with pytest.raises(ValueError):
        d.x = "dog"
