"""Tests for TwoPointFactory re-export from firecrown.likelihood.factories.

This module verifies that TwoPointFactory is accessible from
firecrown.likelihood.factories (for backward compatibility with code written
against older versions of firecrown) and that it is the identical object to
the one available from firecrown.likelihood.
"""

import pytest


def test_two_point_factory_importable():
    """TwoPointFactory can be imported from firecrown.likelihood.factories."""
    # pylint: disable=import-outside-toplevel
    from firecrown.likelihood.factories import TwoPointFactory

    assert TwoPointFactory is not None


def test_two_point_factory_identical_to_likelihood():
    """TwoPointFactory from factories is the same object as from likelihood."""
    # pylint: disable=import-outside-toplevel
    from firecrown.likelihood.factories import TwoPointFactory as FactoriesTWPF
    from firecrown.likelihood import TwoPointFactory

    assert FactoriesTWPF is TwoPointFactory


def test_two_point_factory_in_all():
    """TwoPointFactory is listed in __all__ of firecrown.likelihood.factories."""
    # pylint: disable=import-outside-toplevel
    import firecrown.likelihood.factories as factories_module

    assert "TwoPointFactory" in factories_module.__all__


def test_two_point_factory_is_pydantic_model():
    """TwoPointFactory is a Pydantic BaseModel subclass."""
    # pylint: disable=import-outside-toplevel
    from pydantic import BaseModel
    from firecrown.likelihood.factories import TwoPointFactory

    assert issubclass(TwoPointFactory, BaseModel)


def test_two_point_factory_has_correlation_space_field():
    """TwoPointFactory exposes the correlation_space field."""
    # pylint: disable=import-outside-toplevel
    from firecrown.likelihood.factories import TwoPointFactory

    assert "correlation_space" in TwoPointFactory.model_fields
