"""Tests for deprecated ModelingTools constructor parameters.

This module verifies that the deprecated cluster_abundance and
cluster_deltasigma keyword arguments of ModelingTools.__init__ are
accepted with appropriate deprecation warnings, preserving backward
compatibility with code written against older versions of firecrown.
"""

import warnings

import pytest

from firecrown.modeling_tools import ModelingTools


def test_no_deprecated_args_no_warning():
    """Verify no DeprecationWarning is emitted without deprecated args."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ModelingTools()
    deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecations) == 0


def test_cluster_abundance_none_warns():
    """Passing cluster_abundance=None emits a DeprecationWarning."""
    with pytest.warns(
        DeprecationWarning,
        match="cluster_abundance parameter of ModelingTools is deprecated",
    ):
        ModelingTools(cluster_abundance=None)


def test_cluster_deltasigma_none_warns():
    """Passing cluster_deltasigma=None emits a DeprecationWarning."""
    with pytest.warns(
        DeprecationWarning,
        match="cluster_deltasigma parameter of ModelingTools is deprecated",
    ):
        ModelingTools(cluster_deltasigma=None)


def test_both_deprecated_args_warn():
    """Passing both deprecated args emits two DeprecationWarnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ModelingTools(cluster_abundance=None, cluster_deltasigma=None)

    deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
    messages = [str(x.message) for x in deprecations]
    assert any("cluster_abundance" in m for m in messages)
    assert any("cluster_deltasigma" in m for m in messages)


def test_modeling_tools_still_functional_with_deprecated_args():
    """Verify ModelingTools is functional when constructed with deprecated args."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tools = ModelingTools(cluster_abundance=None, cluster_deltasigma=None)
    assert tools is not None
    assert tools.pt_calculator is None
    assert tools.hm_calculator is None
