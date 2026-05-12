"""Unit tests for the interpolation module (ApplyInterpolationWhen enum)."""

from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from firecrown.models.two_point import ApplyInterpolationWhen


class TestModel(BaseModel):
    """Test model with ApplyInterpolationWhen field."""

    interp: ApplyInterpolationWhen


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
    assert len(all_values) >= 3
    assert ApplyInterpolationWhen.REAL in all_values
    assert ApplyInterpolationWhen.HARMONIC in all_values
    assert ApplyInterpolationWhen.HARMONIC_WINDOW in all_values


def test_pydantic_model_with_apply_interpolation_when():
    """Test using ApplyInterpolationWhen in a Pydantic model."""
    model1 = TestModel(interp=ApplyInterpolationWhen.REAL)
    assert model1.interp == ApplyInterpolationWhen.REAL

    model2 = TestModel.model_validate({"interp": "HARMONIC"})
    assert model2.interp == ApplyInterpolationWhen.HARMONIC

    model3 = TestModel.model_validate({"interp": "REAL | HARMONIC"})
    assert model3.interp & ApplyInterpolationWhen.REAL
    assert model3.interp & ApplyInterpolationWhen.HARMONIC

    model4 = TestModel.model_validate({"interp": "NONE"})
    assert model4.interp == ApplyInterpolationWhen.NONE


def test_pydantic_serialization_with_model():
    """Test serialization of ApplyInterpolationWhen in Pydantic model."""
    model1 = TestModel(interp=ApplyInterpolationWhen.NONE)
    data1 = model1.model_dump()
    assert data1["interp"] == "NONE"

    model2 = TestModel(interp=ApplyInterpolationWhen.REAL)
    data2 = model2.model_dump()
    assert data2["interp"] == "REAL"

    combined = ApplyInterpolationWhen.REAL | ApplyInterpolationWhen.HARMONIC
    model3 = TestModel(interp=combined)
    data3 = model3.model_dump()
    assert "REAL" in data3["interp"]
    assert "HARMONIC" in data3["interp"]


def test_pydantic_validation_errors():
    """Test that invalid inputs raise proper validation errors."""
    with pytest.raises(ValidationError):
        TestModel.model_validate({"interp": "INVALID_FLAG"})

    with pytest.raises((ValidationError, TypeError)):
        TestModel.model_validate(cast(Any, {"interp": 123}))
