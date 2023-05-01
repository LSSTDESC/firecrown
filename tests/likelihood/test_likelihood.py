"""
Tests for the module firecrown.likelihood.likelihood.
"""
import os


from firecrown.likelihood.likelihood import load_likelihood, NamedParameters


def test_load_likelihood_submodule():
    """The likelihood script should be able to load other modules from its
    directory using relative import."""

    dir_path = os.path.dirname(os.path.realpath(__file__))

    load_likelihood(os.path.join(dir_path, "lkdir/lkscript.py"), NamedParameters())
