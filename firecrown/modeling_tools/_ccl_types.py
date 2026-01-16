"""Type definitions for CCL factory module."""

import numpy as np
import numpy.typing as npt

# To be moved to the import from typing when migrating to Python 3.11
from typing_extensions import NotRequired, TypedDict

# PowerSpec is a type that represents a power spectrum.
PowerSpec = TypedDict(
    "PowerSpec",
    {
        "a": npt.NDArray[np.float64],
        "k": npt.NDArray[np.float64],
        "delta_matter:delta_matter": npt.NDArray[np.float64],
    },
)


# Background is a type that represents the cosmological background quantities.
class Background(TypedDict):
    """Type representing cosmological background quantities.

    Contains arrays for scale factor, comoving distance, and Hubble parameter ratio.
    """

    a: npt.NDArray[np.float64]
    chi: npt.NDArray[np.float64]
    h_over_h0: npt.NDArray[np.float64]


# CCLCalculatorArgs is a type that represents the arguments for the
# CCLCalculator.
class CCLCalculatorArgs(TypedDict):
    """Arguments for the CCLCalculator.

    Contains background cosmology and optional linear/nonlinear power spectra.
    """

    background: Background
    pk_linear: NotRequired[PowerSpec]
    pk_nonlin: NotRequired[PowerSpec]
