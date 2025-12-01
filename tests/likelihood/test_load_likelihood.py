"""
Tests for the module firecrown.likelihood.likelihood.
"""

import os
import sys
import pytest

from firecrown.likelihood._likelihood import (
    load_likelihood_from_script,
    load_likelihood_from_module,
    load_likelihood,
    NamedParameters,
)


def test_load_likelihood_from_script_submodule():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    load_likelihood_from_script(
        os.path.join(dir_path, "lkdir/lkscript.py"), NamedParameters()
    )


def test_load_likelihood_from_script_submodule_invalid():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(ValueError, match="Unrecognized Firecrown initialization file"):
        load_likelihood_from_script(
            os.path.join(dir_path, "lkdir/lkscript_invalid.ext"), NamedParameters()
        )


def test_load_likelihood_from_script_submodule_no_build_likelihood():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        AttributeError, match="does not define a `build_likelihood` factory function."
    ):
        load_likelihood_from_script(
            os.path.join(dir_path, "lkdir/lkscript_invalid.py"), NamedParameters()
        )


def test_load_likelihood_from_script_submodule_not_a_function():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        TypeError, match="The factory function `build_likelihood` must be a callable."
    ):
        load_likelihood_from_script(
            os.path.join(dir_path, "lkdir/lkscript_not_a_function.py"),
            NamedParameters(),
        )


def test_load_likelihood_from_script_submodule_returns_wrong_type():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        TypeError,
        match="The returned likelihood must be a Firecrown's `Likelihood` type,",
    ):
        load_likelihood_from_script(
            os.path.join(dir_path, "lkdir/lkscript_returns_wrong_type.py"),
            NamedParameters(),
        )


def test_load_likelihood_from_script_submodule_returns_wrong_type_tools():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.raises(
        TypeError,
        match="The returned tools must be a Firecrown's `ModelingTools` type",
    ):
        load_likelihood_from_script(
            os.path.join(dir_path, "lkdir/lkscript_returns_wrong_type_tools.py"),
            NamedParameters(),
        )


def test_load_likelihood_from_script_submodule_old():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with pytest.deprecated_call():
        load_likelihood_from_script(
            os.path.join(dir_path, "lkdir/lkscript_old.py"),
            NamedParameters(),
        )


def test_load_likelihood_from_script_correct_tools():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _, tools = load_likelihood_from_script(
        os.path.join(dir_path, "lkdir/lkscript.py"), NamedParameters()
    )

    assert tools.test_attribute == "test"  # type: ignore


def test_load_likelihood():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _, tools = load_likelihood(
        os.path.join(dir_path, "lkdir/lkscript.py"), NamedParameters()
    )

    assert tools.test_attribute == "test"  # type: ignore


def test_load_likelihood_from_module():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    module_path = os.path.join(dir_path, "lkdir")

    sys.path.append(module_path)

    _, tools = load_likelihood("lkscript", NamedParameters())

    assert tools.test_attribute == "test"  # type: ignore


def test_load_likelihood_from_module_not_in_path():
    with pytest.raises(
        ValueError,
        match="Unrecognized Firecrown initialization module 'lkscript_not_in_path'. "
        "The module must be either a module_name or a module_name.func where func "
        "is the factory function.",
    ):
        _ = load_likelihood_from_module("lkscript_not_in_path", NamedParameters())


def test_load_likelihood_not_in_path():
    with pytest.raises(
        ValueError,
        match="Unrecognized Firecrown initialization file or module "
        "lkscript_not_in_path.",
    ):
        _ = load_likelihood("lkscript_not_in_path", NamedParameters())


def test_load_likelihood_from_module_function_missing_likelihood_config():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    module_path = os.path.join(dir_path, "lkdir")

    sys.path.append(module_path)

    with pytest.raises(KeyError, match="likelihood_config"):
        _ = load_likelihood_from_module(
            "firecrown.likelihood.factories.build_two_point_likelihood",
            NamedParameters(),
        )
