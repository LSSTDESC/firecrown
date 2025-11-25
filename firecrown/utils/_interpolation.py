"""Interpolation utilities for Firecrown."""

from collections.abc import Callable

import numpy as np
import scipy.interpolate
from numpy import typing as npt


def make_log_interpolator(
    x: npt.NDArray[np.int64], y: npt.NDArray[np.float64]
) -> Callable[[npt.NDArray[np.int64]], npt.NDArray[np.float64]]:
    """Return a function object that does 1D spline interpolation.

    If all the y values are greater than 0, the function
    interpolates log(y) as a function of log(x).
    Otherwise, the function interpolates y as a function of log(x).
    The resulting interpolater will not extrapolate; if called with
    an out-of-range argument it will raise a ValueError.
    """
    if np.all(y > 0):
        # use log-log interpolation
        intp = scipy.interpolate.InterpolatedUnivariateSpline(
            np.log(x), np.log(y), ext=2
        )

        def log_log_interpolator(x_: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
            """Interpolate on log-log scale."""
            return np.exp(intp(np.log(x_)))

        return log_log_interpolator
    # only use log for x
    intp = scipy.interpolate.InterpolatedUnivariateSpline(np.log(x), y, ext=2)

    def log_x_interpolator(x_: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Interpolate on log-x scale."""
        return intp(np.log(x_))

    return log_x_interpolator
