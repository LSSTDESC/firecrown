"""
Tests to ensure that oldmodule imports still work, and yield modules that contain the
same names as the newmodule imports.
"""

# pylint: disable=import-outside-toplevel

import re
import types
import pytest

HIDDEN_NAME_PATTERN = re.compile(r"_\w+")


def diff_module_names(
    oldmodule: types.ModuleType, newmodule: types.ModuleType
) -> set[str]:
    """Return a set containing the names specified by the given module."""
    old_names = set(x for x in dir(oldmodule) if not HIDDEN_NAME_PATTERN.match(x))
    new_names = set(x for x in dir(newmodule) if not HIDDEN_NAME_PATTERN.match(x))

    return old_names ^ new_names


def test_import_binned_cluster_number_counts():
    with pytest.deprecated_call():
        import firecrown.likelihood.gauss_family.statistic.binned_cluster_number_counts as oldmodule  # pylint: disable=line-too-long # noqa: E501
        import firecrown.likelihood.binned_cluster_number_counts as newmodule

        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])


def test_import_gaussian():
    with pytest.deprecated_call():
        import firecrown.likelihood.gauss_family.gaussian as oldmodule
        import firecrown.likelihood.gaussian as newmodule

        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])


def test_import_number_counts():
    with pytest.deprecated_call():
        from firecrown.likelihood.gauss_family.statistic.source import (
            number_counts as oldmodule,
        )
        import firecrown.likelihood.number_counts as newmodule

        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])


def test_import_source():
    with pytest.deprecated_call():
        import firecrown.likelihood.gauss_family.statistic.source.source as oldmodule
        import firecrown.likelihood.source as newmodule

        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])


def test_import_statistic():
    with pytest.deprecated_call():
        import firecrown.likelihood.gauss_family.statistic.statistic as oldmodule
        import firecrown.likelihood.statistic as newmodule

        # The statistic module contains warnings.
        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])


def test_import_student_t():
    with pytest.deprecated_call():
        import firecrown.likelihood.gauss_family.student_t as oldmodule
        import firecrown.likelihood.student_t as newmodule

        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])


def test_import_two_point():
    with pytest.deprecated_call():
        import firecrown.likelihood.gauss_family.statistic.two_point as oldmodule
        import firecrown.likelihood.two_point as newmodule

        # The two_point module contains warnings.
        assert diff_module_names(oldmodule, newmodule) == set()


def test_import_supernova():
    with pytest.deprecated_call():
        import firecrown.likelihood.gauss_family.statistic.supernova as oldmodule
        import firecrown.likelihood.supernova as newmodule

        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])


def test_import_weak_lensing():
    with pytest.deprecated_call():
        from firecrown.likelihood.gauss_family.statistic.source import (
            weak_lensing as oldmodule,
        )
        import firecrown.likelihood.weak_lensing as newmodule

        assert diff_module_names(oldmodule, newmodule) == set(["warnings"])
