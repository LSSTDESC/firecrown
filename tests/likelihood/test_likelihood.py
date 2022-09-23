import pytest
import os

from firecrown.likelihood.likelihood import load_likelihood


def test_load_likelihood_submodule():
    """The likelihood script shoult be able to load other modules from its
    directory using relative import."""

    dir_path = os.path.dirname(os.path.realpath(__file__))

    load_likelihood(os.path.join(dir_path, "lkdir/lkscript.py"))
