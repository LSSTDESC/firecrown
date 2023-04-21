"""
Provides a trivial likelihood factory function for testing purposes.
"""
from . import lkmodule


def build_likelihood(_):
    """Return an EmptyLikelihood object."""
    return lkmodule.empty_likelihood()
