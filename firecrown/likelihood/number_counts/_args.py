"""Arguments for number counts tracers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from firecrown._source_arrays import (
    _DndzBit,
    _NBit,
    _RealArray,
    _as_float64,
    _as_float64_pair,
)
from firecrown.likelihood_base import SourceGalaxyArgs


@dataclass(frozen=True, init=False)
class NumberCountsArgs(SourceGalaxyArgs):
    """Class for number counts tracer builder argument."""

    bias: None | npt.NDArray[np.float64] = None
    mag_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    has_pt: bool = False
    has_hm: bool = False
    b_2: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    b_s: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None

    # Preserve the existing positional dataclass signature with flexible z/dndz.
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        z: _RealArray[_NBit],
        dndz: _RealArray[_DndzBit],
        scale: float = 1.0,
        field: str = "delta_matter",
        bias: None | npt.NDArray[np.float64] = None,
        mag_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
        has_pt: bool = False,
        has_hm: bool = False,
        b_2: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
        b_s: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
    ) -> None:
        """Build source arguments with float64 redshift storage.

        :param z: Real integer or floating redshift coordinates.
        :param dndz: Real integer or floating distribution values.
        :param scale: Source scale.
        :param field: Power-spectrum field.
        :param bias: See the corresponding source argument field.
        :param mag_bias: See the corresponding source argument field.
        :param has_pt: See the corresponding source argument field.
        :param has_hm: See the corresponding source argument field.
        :param b_2: See the corresponding source argument field.
        :param b_s: See the corresponding source argument field.
        """
        super().__init__(z=z, dndz=dndz, scale=scale, field=field)
        object.__setattr__(self, "bias", None if bias is None else _as_float64(bias))
        object.__setattr__(self, "mag_bias", _as_float64_pair(mag_bias))
        object.__setattr__(self, "has_pt", has_pt)
        object.__setattr__(self, "has_hm", has_hm)
        object.__setattr__(self, "b_2", _as_float64_pair(b_2))
        object.__setattr__(self, "b_s", _as_float64_pair(b_s))
