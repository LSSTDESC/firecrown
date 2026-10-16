"""
Provides a trivial likelihood factory function for testing purposes.
This module should be loaded by the test_load_likelihood_submodule test.
It should raise an exception because the factory function does not define
a build_likelihood Callable.
"""

from firecrown.likelihood._likelihood import NamedParameters

from . import lkmodule


def not_a_build_likelihood(_: NamedParameters) -> lkmodule.EmptyLikelihood:
    """Return an EmptyLikelihood object."""
    return lkmodule.empty_likelihood()
