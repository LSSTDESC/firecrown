"""Arguments for number counts tracers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from firecrown.likelihood._base import SourceGalaxyArgs


@dataclass(frozen=True)
class NumberCountsArgs(SourceGalaxyArgs):
    """Class for number counts tracer builder argument."""

    bias: None | npt.NDArray[np.float64] = None
    mag_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    has_pt: bool = False
    has_hm: bool = False
    b_2: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    b_s: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
