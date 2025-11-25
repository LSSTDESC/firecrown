"""Tests for GaussFamily base class."""

import re

import pytest
import firecrown.likelihood._gaussian as g


def test_init_rejects_non_statistics():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "statistics[0] is not an instance of Statistic. It is a <class 'int'>."
        ),
    ):
        g.ConstGaussian([1])  # type: ignore
