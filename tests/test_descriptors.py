import pytest
import math
from firecrown.descriptors import TypeFloat, TypeString, TypeLikelihood
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian


class NotALikelihood:
    """A trivial class that does not inherit from Likelihood."""

    pass


class DescriptorTester:
    unconstrained_optional_float = TypeFloat(allow_none=True)
    unconstrained_required_float = TypeFloat()
    constrained_optional_float = TypeFloat(1.0, 2.5, allow_none=True)
    constrained_required_float = TypeFloat(10.0, 100.0)


class HasUOF:
    """Test harness for unconstrained optional float"""

    x = TypeFloat(allow_none=True)

    def __init__(self):
        self.x = None


class HasCOF:
    """Test harness for constrained optional float"""

    x = TypeFloat(1.0, 3.0, allow_none=True)

    def __init__(self):
        self.x = None


class HasURF:
    """Test harness for unconstrained required float"""

    x = TypeFloat()

    def __init__(self):
        self.x = 0.0


class HasCRF:
    """Test harness for constrained required float"""

    x = TypeFloat(1.0, 3.0)

    def __init__(self):
        self.x = 2.0


class HasLowerBound:
    """Test harness for float with only a lower bound"""

    x = TypeFloat(minvalue=0.0)

    def __init__(self):
        self.x = 0.0


class HasUpperBound:
    """Test harness for float with only an upper bound"""

    x = TypeFloat(maxvalue=1.0)

    def __init__(self):
        self.x = 1.0


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
    assert d.x is 2.0

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
