"""Tests for GaussFamily base class."""

import re

import pytest
from firecrown.likelihood import ConstGaussian


def test_init_rejects_non_statistics():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "statistics[0] is not an instance of Statistic. It is a <class 'int'>."
        ),
    ):
        ConstGaussian([1])  # type: ignore[list-item]
