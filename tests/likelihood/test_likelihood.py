"""
Tests for the module firecrown.likelihood.likelihood.
"""
import os
import pytest

from firecrown.likelihood.likelihood import load_likelihood, NamedParameters


def test_load_likelihood_submodule():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    load_likelihood(os.path.join(dir_path, "lkdir/lkscript.py"), NamedParameters())


def test_load_likelihood_submodule_invalid():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(ValueError, match="Unrecognized Firecrown initialization file"):
        load_likelihood(
            os.path.join(dir_path, "lkdir/lkscript_invalid.ext"), NamedParameters()
        )


def test_load_likelihood_submodule_no_build_likelihood():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        AttributeError, match="does not define a `build_likelihood` factory function."
    ):
        load_likelihood(
            os.path.join(dir_path, "lkdir/lkscript_invalid.py"), NamedParameters()
        )


def test_load_likelihood_submodule_not_a_function():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        TypeError, match="The factory function `build_likelihood` must be a callable."
    ):
        load_likelihood(
            os.path.join(dir_path, "lkdir/lkscript_not_a_function.py"),
            NamedParameters(),
        )


def test_load_likelihood_submodule_returns_wrong_type():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        TypeError,
        match="The returned likelihood must be a Firecrown's `Likelihood` type,",
    ):
        load_likelihood(
            os.path.join(dir_path, "lkdir/lkscript_returns_wrong_type.py"),
            NamedParameters(),
        )


def test_load_likelihood_submodule_returns_wrong_type_tools():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        TypeError,
        match="The returned tools must be a Firecrown's `ModelingTools` type",
    ):
        load_likelihood(
            os.path.join(dir_path, "lkdir/lkscript_returns_wrong_type_tools.py"),
            NamedParameters(),
        )


def test_load_likelihood_submodule_old():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    load_likelihood(
        os.path.join(dir_path, "lkdir/lkscript_old.py"),
        NamedParameters(),
    )


def test_load_likelihood_correct_tools():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _, tools = load_likelihood(
        os.path.join(dir_path, "lkdir/lkscript.py"), NamedParameters()
    )

    assert tools.test_attribute == "test"  # type: ignore
