"""Unit tests for the interpolation module (ApplyInterpolationWhen enum)."""

import pytest
from pydantic import BaseModel, ValidationError

from firecrown.models.two_point import ApplyInterpolationWhen


def test_apply_interpolation_when_none():
    """Test ApplyInterpolationWhen.NONE flag."""
    apply_interp_when = ApplyInterpolationWhen.NONE
    assert apply_interp_when == ApplyInterpolationWhen.NONE
    assert not apply_interp_when & ApplyInterpolationWhen.REAL
    assert not apply_interp_when & ApplyInterpolationWhen.HARMONIC
    assert not apply_interp_when & ApplyInterpolationWhen.HARMONIC_WINDOW


def test_apply_interpolation_when_all():
    """Test ApplyInterpolationWhen.ALL flag."""
    apply_interp_when = ApplyInterpolationWhen.ALL
    assert apply_interp_when == ApplyInterpolationWhen.ALL
    assert apply_interp_when & ApplyInterpolationWhen.REAL
    assert apply_interp_when & ApplyInterpolationWhen.HARMONIC
    assert apply_interp_when & ApplyInterpolationWhen.HARMONIC_WINDOW


def test_apply_interpolation_when_default():
    """Test ApplyInterpolationWhen.DEFAULT flag composition."""
    apply_interp_when = ApplyInterpolationWhen.DEFAULT
    assert apply_interp_when == ApplyInterpolationWhen.DEFAULT
    assert apply_interp_when & ApplyInterpolationWhen.REAL
    assert apply_interp_when & ApplyInterpolationWhen.HARMONIC_WINDOW
    assert not apply_interp_when & ApplyInterpolationWhen.HARMONIC


def test_apply_interpolation_when_real():
    """Test ApplyInterpolationWhen.REAL flag."""
    apply_interp_when = ApplyInterpolationWhen.REAL
    assert apply_interp_when & ApplyInterpolationWhen.REAL
    assert not apply_interp_when & ApplyInterpolationWhen.HARMONIC
    assert not apply_interp_when & ApplyInterpolationWhen.HARMONIC_WINDOW


def test_apply_interpolation_when_harmonic():
    """Test ApplyInterpolationWhen.HARMONIC flag."""
    apply_interp_when = ApplyInterpolationWhen.HARMONIC
    assert not apply_interp_when & ApplyInterpolationWhen.REAL
    assert apply_interp_when & ApplyInterpolationWhen.HARMONIC
    assert not apply_interp_when & ApplyInterpolationWhen.HARMONIC_WINDOW


def test_apply_interpolation_when_harmonic_window():
    """Test ApplyInterpolationWhen.HARMONIC_WINDOW flag."""
    apply_interp_when = ApplyInterpolationWhen.HARMONIC_WINDOW
    assert not apply_interp_when & ApplyInterpolationWhen.REAL
    assert not apply_interp_when & ApplyInterpolationWhen.HARMONIC
    assert apply_interp_when & ApplyInterpolationWhen.HARMONIC_WINDOW


def test_apply_interpolation_when_bitwise_or():
    """Test bitwise OR operations on ApplyInterpolationWhen flags."""
    combined = ApplyInterpolationWhen.REAL | ApplyInterpolationWhen.HARMONIC
    assert combined & ApplyInterpolationWhen.REAL
    assert combined & ApplyInterpolationWhen.HARMONIC
    assert not combined & ApplyInterpolationWhen.HARMONIC_WINDOW


def test_apply_interpolation_when_iteration():
    """Test that all ApplyInterpolationWhen values are iterable."""
    all_values = list(ApplyInterpolationWhen)
    assert len(all_values) >= 3  # At minimum: REAL, HARMONIC, HARMONIC_WINDOW
    assert ApplyInterpolationWhen.REAL in all_values
    assert ApplyInterpolationWhen.HARMONIC in all_values
    assert ApplyInterpolationWhen.HARMONIC_WINDOW in all_values


def test_pydantic_schema_exists():
    """Test that Pydantic core schema can be retrieved."""
    result = ApplyInterpolationWhen.__get_pydantic_core_schema__(
        ApplyInterpolationWhen, None  # type: ignore[arg-type]
    )
    assert result is not None


def test_pydantic_model_with_apply_interpolation_when():
    """Test using ApplyInterpolationWhen in a Pydantic model."""

    class TestModel(BaseModel):
        """Test model with ApplyInterpolationWhen field."""

        interp: ApplyInterpolationWhen

    # Test with instance
    model1 = TestModel(interp=ApplyInterpolationWhen.REAL)
    assert model1.interp == ApplyInterpolationWhen.REAL

    # Test with string
    model2 = TestModel(interp="HARMONIC")  # type: ignore[arg-type]
    assert model2.interp == ApplyInterpolationWhen.HARMONIC

    # Test with multiple flags
    model3 = TestModel(interp="REAL | HARMONIC")  # type: ignore[arg-type]
    assert model3.interp & ApplyInterpolationWhen.REAL
    assert model3.interp & ApplyInterpolationWhen.HARMONIC

    # Test with NONE
    model4 = TestModel(interp="NONE")  # type: ignore[arg-type]
    assert model4.interp == ApplyInterpolationWhen.NONE


def test_pydantic_serialization_with_model():
    """Test serialization of ApplyInterpolationWhen in Pydantic model."""

    class TestModel(BaseModel):
        """Test model with ApplyInterpolationWhen field."""

        interp: ApplyInterpolationWhen

    # Test serializing NONE
    model1 = TestModel(interp=ApplyInterpolationWhen.NONE)
    data1 = model1.model_dump()
    assert data1["interp"] == "NONE"

    # Test serializing single flag
    model2 = TestModel(interp=ApplyInterpolationWhen.REAL)
    data2 = model2.model_dump()
    assert "REAL" in data2["interp"]

    # Test serializing combined flags
    combined = ApplyInterpolationWhen.REAL | ApplyInterpolationWhen.HARMONIC
    model3 = TestModel(interp=combined)
    data3 = model3.model_dump()
    assert "REAL" in data3["interp"]
    assert "HARMONIC" in data3["interp"]


def test_pydantic_validation_errors():
    """Test that invalid inputs raise proper validation errors."""

    class TestModel(BaseModel):
        """Test model with ApplyInterpolationWhen field."""

        interp: ApplyInterpolationWhen

    # Test invalid flag name
    with pytest.raises(ValidationError):
        TestModel(interp="INVALID_FLAG")  # type: ignore[arg-type]

    # Test invalid type - TypeError gets wrapped in ValidationError by Pydantic
    with pytest.raises((ValidationError, TypeError)):
        TestModel(interp=123)  # type: ignore[arg-type]


def test_pydantic_serialize_none():
    """Test Pydantic serialization of NONE flag."""
    schema = ApplyInterpolationWhen.__get_pydantic_core_schema__(
        ApplyInterpolationWhen, None  # type: ignore[arg-type]
    )
    # Get the serialization function from the schema
    serialization = schema["serialization"]
    serialize_func = serialization["function"]

    result = serialize_func(ApplyInterpolationWhen.NONE)
    assert result == "NONE"


def test_pydantic_serialize_single_flag():
    """Test Pydantic serialization of a single flag."""
    schema = ApplyInterpolationWhen.__get_pydantic_core_schema__(
        ApplyInterpolationWhen, None  # type: ignore[arg-type]
    )
    serialization = schema["serialization"]
    serialize_func = serialization["function"]

    result = serialize_func(ApplyInterpolationWhen.REAL)
    assert result == "REAL"


def test_pydantic_serialize_multiple_flags():
    """Test Pydantic serialization of combined flags."""
    schema = ApplyInterpolationWhen.__get_pydantic_core_schema__(
        ApplyInterpolationWhen, None  # type: ignore[arg-type]
    )
    serialization = schema["serialization"]
    serialize_func = serialization["function"]

    combined = ApplyInterpolationWhen.REAL | ApplyInterpolationWhen.HARMONIC
    result = serialize_func(combined)

    # Result should contain both flags separated by pipe
    assert "REAL" in result
    assert "HARMONIC" in result
    assert "|" in result


def test_pydantic_serialize_default():
    """Test Pydantic serialization of DEFAULT flag."""
    schema = ApplyInterpolationWhen.__get_pydantic_core_schema__(
        ApplyInterpolationWhen, None  # type: ignore[arg-type]
    )
    serialization = schema["serialization"]
    serialize_func = serialization["function"]

    result = serialize_func(ApplyInterpolationWhen.DEFAULT)

    # DEFAULT = REAL | HARMONIC_WINDOW
    assert "REAL" in result
    assert "HARMONIC_WINDOW" in result
