"""
Tests for the firecrown.utils modle.
"""

import pytest
import numpy as np

from firecrown.utils import (
    upper_triangle_indices,
    save_to_sacc,
    compare_optional_arrays,
    compare_optionals,
    base_model_from_yaml,
)


def test_upper_triangle_indices_nonzero():
    indices = list(upper_triangle_indices(3))
    assert indices == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_upper_triangle_indices_zero():
    indices = list(upper_triangle_indices(0))
    assert not indices


def test_save_to_sacc(trivial_stats, sacc_data_for_trivial_stat):
    stat = trivial_stats[0]
    stat.read(sacc_data_for_trivial_stat)
    idx = np.arange(stat.count)
    new_data_vector = 3 * stat.get_data_vector()[idx]

    new_sacc = save_to_sacc(
        sacc_data=sacc_data_for_trivial_stat,
        data_vector=new_data_vector,
        indices=idx,
        strict=True,
    )
    assert all(new_sacc.data[i].value == d for i, d in zip(idx, new_data_vector))


def test_save_to_sacc_strict_fail(trivial_stats, sacc_data_for_trivial_stat):
    stat = trivial_stats[0]
    stat.read(sacc_data_for_trivial_stat)
    idx = np.arange(stat.count - 1)
    new_data_vector = stat.get_data_vector()[idx]

    with pytest.raises(RuntimeError):
        _ = save_to_sacc(
            sacc_data=sacc_data_for_trivial_stat,
            data_vector=new_data_vector,
            indices=idx,
            strict=True,
        )


def test_compare_optional_arrays_():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    assert compare_optional_arrays(x, y)

    z = np.array([1, 2, 4])
    assert not compare_optional_arrays(x, z)

    a = None
    b = None
    assert compare_optional_arrays(a, b)

    q = np.array([1, 2, 3])
    assert not compare_optional_arrays(q, a)

    assert not compare_optional_arrays(a, q)


def test_base_model_from_yaml_wrong():
    with pytest.raises(ValueError):
        _ = base_model_from_yaml(str, "wrong")


def test_compare_optionals():
    x = "test"
    y = "test"
    assert compare_optionals(x, y)

    z = "test2"
    assert not compare_optionals(x, z)

    a = None
    b = None
    assert compare_optionals(a, b)

    q = np.array([1, 2, 3])
    assert not compare_optionals(q, a)

    assert not compare_optionals(a, q)
