"""Generator support for TwoPoint statistics."""

from __future__ import annotations

from typing import Annotated, TypedDict

from pydantic import BaseModel, Field, model_validator, ConfigDict
import numpy as np
import numpy.typing as npt


class LogLinearElls(BaseModel):
    """Generator for log-linear integral ell values.

    The initializer for LogLinearElls accepts only named parameters. A default
    value is provided for each parameter, which default is used if the parameter
    is not provided to the initializer.

    Not all ell values will be generated. The result will contain each integral
    value from min to mid. Starting from mid, and going up to max, there will be
    n_log logarithmically spaced values.

    Note that midpoint must be strictly greater than minimum, and strictly less
    than maximum. n_log must be positive. All must be integers. All these conditions
    are verified in the initializer.

    LogLinearElls objects are immutable, so they are safe to share and to use as
    default values for parameters.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: Annotated[int, Field(ge=0)] = 2
    midpoint: Annotated[int, Field(ge=0)] = 50
    maximum: Annotated[int, Field(ge=0)] = 60_000
    n_log: Annotated[int, Field(ge=1)] = 200

    @model_validator(mode="after")
    def require_increasing(self) -> "LogLinearElls":
        """Validate the ell values."""
        assert self.minimum < self.midpoint
        assert self.midpoint < self.maximum
        return self

    def generate(
        self, min_ell: int | None = None, max_ell: int | None = None
    ) -> npt.NDArray[np.int64]:
        """Generate the log-linear ell values.

        This will use the object's minimum, midpoint, maximum, and n_log,
        unless min_ell and max_ell are provided, in which case those will be used.

        :param min_ell: The low edge of the first bin.
        :param max_ell: The high edge of the last bin.
        :return: The ell values.
        """
        minimum, midpoint, maximum, n_log = (
            self.minimum if min_ell is None else max(min_ell, self.minimum),
            self.midpoint,
            self.maximum if max_ell is None else max_ell,
            self.n_log,
        )
        assert minimum < midpoint
        assert midpoint < maximum

        lower_range = np.linspace(minimum, midpoint - 1, midpoint - minimum)
        upper_range = np.logspace(np.log10(midpoint), np.log10(maximum), n_log)
        concatenated = np.concatenate((lower_range, upper_range))
        # Round the results to the nearest integer values.
        # N.B. the dtype of the result is np.dtype[float64]
        return np.unique(np.around(concatenated)).astype(np.int64)


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
            return np.array((edges[1:] + edges[:-1]) / 2.0, dtype=np.float64)
        case _:
            raise ValueError(f"Unrecognized binning: {binning}")


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


def generate_ells_cells(ell_config: EllOrThetaConfig):
    """Generate ells or theta values from the configuration dictionary.

    :param ell_config: the configuration parameters.
    :return: ells and Cells
    """
    ells = generate_bin_centers(**ell_config)
    Cells = np.zeros_like(ells)

    return ells, Cells


def generate_reals(theta_config: EllOrThetaConfig):
    """Generate theta and xi values from the configuration dictionary.

    :param ell_config: the configuration parameters.
    :return: ells and Cells
    """
    thetas = generate_bin_centers(**theta_config)
    xis = np.zeros_like(thetas)

    return thetas, xis
