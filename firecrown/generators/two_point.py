"""Generator support for TwoPoint statistics."""

from typing import Annotated
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
