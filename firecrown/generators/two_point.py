"""Generator support for TwoPoint statistics."""

from __future__ import annotations

import copy
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field, model_validator
import numpy as np
import numpy.typing as npt


ELL_FOR_XI_DEFAULTS = {"minimum": 2, "midpoint": 50, "maximum": 60_000, "n_log": 200}


class LogLinearElls(BaseModel):
    """Generator for log-linear integral ell values.

    Not all ell values will be generated. The result will contain each integral
    value from min to mid. Starting from mid, and going up to max, there will be
    n_log logarithmically spaced values.

    Note that midpoint must be strictly greater than minimum, and strictly less
    than maximum. n_log must be positive.
    """

    minimum: Annotated[int, Field(ge=0)]
    midpoint: Annotated[int, Field(ge=0)]
    maximum: Annotated[int, Field(ge=0)]
    n_log: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def require_increasing(self) -> "LogLinearElls":
        """Validate the ell values."""
        assert self.minimum < self.midpoint
        assert self.midpoint < self.maximum
        return self

    def generate(self) -> npt.NDArray[np.int64]:
        """Generate the log-linear ell values.

        The result will contain each integral value from min to mid. Starting
        from mid, and going up to max, there will be n_log logarithmically
        spaced values.
        """
        minimum, midpoint, maximum, n_log = (
            self.minimum,
            self.midpoint,
            self.maximum,
            self.n_log,
        )
        lower_range = np.linspace(minimum, midpoint - 1, midpoint - minimum)
        upper_range = np.logspace(np.log10(midpoint), np.log10(maximum), n_log)
        concatenated = np.concatenate((lower_range, upper_range))
        # Round the results to the nearest integer values.
        # N.B. the dtype of the result is np.dtype[float64]
        return np.unique(np.around(concatenated)).astype(np.int64)


def log_linear_ells(
    *, minimum: int, midpoint: int, maximum: int, n_log: int
) -> npt.NDArray[np.int64]:
    """Create an array of ells to sample the power spectrum.

    This is used for for real-space predictions. The result will contain
    each integral value from min to mid. Starting from mid, and going up
    to max, there will be n_log logarithmically spaced values.

    All values are rounded to the nearest integer.
    """
    return LogLinearElls(
        minimum=minimum, midpoint=midpoint, maximum=maximum, n_log=n_log
    ).generate()


def generate_bin_centers(
    *, minimum: float, maximum: float, n: int, binning: str = "log"
) -> npt.NDArray[np.float64]:
    """Return the centers of bins that span the range from minimum to maximum.

    If binning is 'log', this will generate logarithmically spaced bins; if
    binning is 'lin', this will generate linearly spaced bins.

    :param minimum: The low edge of the first bin.
    :param maximum: The high edge of the last bin.
    :param n: The number of bins.
    :param binning: Either 'log' or 'lin'.
    :return: The centers of the bins.
    """
    match binning:
        case "log":
            edges = np.logspace(np.log10(minimum), np.log10(maximum), n + 1)
            return np.sqrt(edges[1:] * edges[:-1])
        case "lin":
            edges = np.linspace(minimum, maximum, n + 1)
            return (edges[1:] + edges[:-1]) / 2.0
        case _:
            raise ValueError(f"Unrecognized binning: {binning}")


def calculate_ells_for_interpolation(
    min_ell: int, max_ell: int
) -> npt.NDArray[np.int64]:
    """See log_linear_ells.

    This method mixes together:
        1. the default parameters in ELL_FOR_XI_DEFAULTS
        2. the first and last values in w.

    and then calls log_linear_ells with those arguments, returning whatever it
    returns.
    """
    ell_config = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
    ell_config["maximum"] = max_ell
    ell_config["minimum"] = max(ell_config["minimum"], min_ell)
    return log_linear_ells(**ell_config)


class EllOrThetaConfig(TypedDict):
    """A dictionary of options for generating the ell or theta.

    This dictionary contains the minimum, maximum and number of
    bins to generate the ell or theta values at which to compute the statistics.

    :param minimum: The start of the binning.
    :param maximum: The end of the binning.
    :param n: The number of bins.
    :param binning: Pass 'log' to get logarithmic spaced bins and 'lin' to get linearly
        spaced bins. Default is 'log'.

    """

    minimum: float
    maximum: float
    n: int
    binning: str
