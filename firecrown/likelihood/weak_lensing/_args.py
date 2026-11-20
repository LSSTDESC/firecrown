"""Arguments for weak lensing tracers."""

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

    # Preserve the existing positional dataclass signature with flexible z/dndz.
    # Its parameter count also accounts for the constructor's local variables.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    def __init__(
        self,
        z: _RealArray[_NBit],
        dndz: _RealArray[_DndzBit],
        scale: float = 1.0,
        field: str = "delta_matter",
        ia_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
        ia_amplitude: None | np.float64 = None,
        ia_mass_scaling: None | np.float64 = None,
        red_fraction: None | np.float64 = None,
        log10_average_halo_mass: None | np.float64 = None,
        has_pt: bool = False,
        has_hm: bool = False,
        ia_pt_c_1: (
            None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ) = None,
        ia_pt_c_d: (
            None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ) = None,
        ia_pt_c_2: (
            None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ) = None,
        ia_a_1h: None | npt.NDArray[np.float64] = None,
        ia_a_2h: None | npt.NDArray[np.float64] = None,
    ) -> None:
        """Build source arguments with float64 redshift storage.

        :param z: Real integer or floating redshift coordinates.
        :param dndz: Real integer or floating distribution values.
        :param scale: Source scale.
        :param field: Power-spectrum field.
        :param ia_bias: See the corresponding source argument field.
        :param ia_amplitude: See the corresponding source argument field.
        :param ia_mass_scaling: See the corresponding source argument field.
        :param red_fraction: See the corresponding source argument field.
        :param log10_average_halo_mass: See the corresponding source argument field.
        :param has_pt: See the corresponding source argument field.
        :param has_hm: See the corresponding source argument field.
        :param ia_pt_c_1: See the corresponding source argument field.
        :param ia_pt_c_d: See the corresponding source argument field.
        :param ia_pt_c_2: See the corresponding source argument field.
        :param ia_a_1h: See the corresponding source argument field.
        :param ia_a_2h: See the corresponding source argument field.
        """
        super().__init__(z=z, dndz=dndz, scale=scale, field=field)
        object.__setattr__(self, "ia_bias", _as_float64_pair(ia_bias))
        object.__setattr__(self, "ia_amplitude", ia_amplitude)
        object.__setattr__(self, "ia_mass_scaling", ia_mass_scaling)
        object.__setattr__(self, "red_fraction", red_fraction)
        object.__setattr__(self, "log10_average_halo_mass", log10_average_halo_mass)
        object.__setattr__(self, "has_pt", has_pt)
        object.__setattr__(self, "has_hm", has_hm)
        object.__setattr__(self, "ia_pt_c_1", _as_float64_pair(ia_pt_c_1))
        object.__setattr__(self, "ia_pt_c_d", _as_float64_pair(ia_pt_c_d))
        object.__setattr__(self, "ia_pt_c_2", _as_float64_pair(ia_pt_c_2))
        object.__setattr__(
            self, "ia_a_1h", None if ia_a_1h is None else _as_float64(ia_a_1h)
        )
        object.__setattr__(
            self, "ia_a_2h", None if ia_a_2h is None else _as_float64(ia_a_2h)
        )

    # pylint: enable=too-many-arguments,too-many-positional-arguments,too-many-locals
