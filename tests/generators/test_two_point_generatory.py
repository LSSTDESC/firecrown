"""Tests for the firecrown.generators.two_point module."""

import numpy as np
import numpy.typing as npt
import pytest

from firecrown.generators.two_point import apply_ells_min_max, apply_theta_min_max


@pytest.mark.parametrize(
    "ells, Cells, indices, ell_min, ell_max, res_ells, res_Cells, res_indices",
    [
        (
            np.array([2, 3, 4, 5, 6]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
            None,
            None,
            np.array([2, 3, 4, 5, 6]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
        ),
        (
            np.array([2, 3, 4, 5, 6]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
            3,
            None,
            np.array([3, 4, 5, 6]),
            np.array([2.0, 3.0, 4.0, 5.0]),
            np.array([1, 2, 3, 4]),
        ),
        (
            np.array([2, 3, 4, 5, 6]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
            None,
            5,
            np.array([2, 3, 4, 5]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0, 1, 2, 3]),
        ),
        (
            np.array([2, 3, 4, 5, 6]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            None,
            3,
            5,
            np.array([3, 4, 5]),
            np.array([2.0, 3.0, 4.0]),
            None,
        ),
    ],
)
def test_apply_ells_min_max(
    ells: npt.NDArray[np.int64],
    Cells: npt.NDArray[np.float64],
    indices: None | npt.NDArray[np.int64],
    ell_min: None | int,
    ell_max: None | int,
    res_ells: np.typing.NDArray[np.int64],
    res_Cells: np.typing.NDArray[np.float64],
    res_indices: None | np.typing.NDArray[np.int64],
):
    """Test the apply_ells_min_max function."""
    (
        result_ells,
        result_Cells,
        result_indices,
    ) = apply_ells_min_max(ells, Cells, indices, ell_min, ell_max)
    np.testing.assert_array_equal(result_ells, res_ells)
    np.testing.assert_array_equal(result_Cells, res_Cells)
    if indices is None:
        assert result_indices is None
    else:
        assert result_indices is not None
        assert res_indices is not None
        np.testing.assert_array_equal(result_indices, res_indices)


@pytest.mark.parametrize(
    "thetas, xis, indices, theta_min, theta_max, res_thetas, res_xis, res_indices",
    [
        (
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
            None,
            None,
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
        ),
        (
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
            20.0,
            None,
            np.array([20.0, 30.0, 40.0, 50.0]),
            np.array([2.0, 3.0, 4.0, 5.0]),
            np.array([1, 2, 3, 4]),
        ),
        (
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 1, 2, 3, 4]),
            None,
            40.0,
            np.array([10.0, 20.0, 30.0, 40.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0, 1, 2, 3]),
        ),
        (
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            None,
            20.0,
            40.0,
            np.array([20.0, 30.0, 40.0]),
            np.array([2.0, 3.0, 4.0]),
            None,
        ),
    ],
)
def test_apply_theta_min_max(
    thetas: npt.NDArray[np.float64],
    xis: npt.NDArray[np.float64],
    indices: None | npt.NDArray[np.int64],
    theta_min: None | float,
    theta_max: None | float,
    res_thetas: npt.NDArray[np.float64],
    res_xis: npt.NDArray[np.float64],
    res_indices: None | npt.NDArray[np.int64],
):
    result_thetas, result_xis, result_indices = apply_theta_min_max(
        thetas, xis, indices, theta_min, theta_max
    )

    np.testing.assert_array_equal(result_thetas, res_thetas)
    np.testing.assert_array_equal(result_xis, res_xis)
    if indices is not None:
        assert result_indices is not None
        assert res_indices is not None
        np.testing.assert_array_equal(result_indices, res_indices)
    else:
        assert result_indices is None
