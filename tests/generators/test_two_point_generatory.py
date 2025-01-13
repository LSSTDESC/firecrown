"""Tests for the firecrown.generators.two_point module."""

from firecrown.generators import two_point


import numpy as np
import numpy.typing as npt
import pytest

from firecrown.generators.two_point import apply_ells_min_max


@pytest.mark.parametrize(
    "ells, Cells, indices, ell_min, ell_max, expected_ells, expected_Cells, expected_indices",
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
    expected_ells: np.typing.NDArray[np.int64],
    expected_Cells: np.typing.NDArray[np.float64],
    expected_indices: None | np.typing.NDArray[np.int64],
):
    """Test the apply_ells_min_max function."""
    (
        result_ells,
        result_Cells,
        result_indices,
    ) = apply_ells_min_max(ells, Cells, indices, ell_min, ell_max)
    np.testing.assert_array_equal(result_ells, expected_ells)
    np.testing.assert_array_equal(result_Cells, expected_Cells)
    np.testing.assert_array_equal(result_indices, expected_indices)
