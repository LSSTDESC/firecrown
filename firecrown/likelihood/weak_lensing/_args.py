"""Arguments for weak lensing tracers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from firecrown.likelihood_base import SourceGalaxyArgs


@dataclass(frozen=True)
class WeakLensingArgs(SourceGalaxyArgs):
    """Class for weak lensing tracer builder argument."""

    ia_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None

    ia_amplitude: None | np.float64 = None
    ia_mass_scaling: None | np.float64 = None
    red_fraction: None | np.float64 = None
    log10_average_halo_mass: None | np.float64 = None

    has_pt: bool = False
    has_hm: bool = False

    ia_pt_c_1: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    ia_pt_c_d: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    ia_pt_c_2: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None

    ia_a_1h: None | npt.NDArray[np.float64] = None
    ia_a_2h: None | npt.NDArray[np.float64] = None
