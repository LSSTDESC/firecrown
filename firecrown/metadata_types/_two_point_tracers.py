"""Tomographic bin / tracer types.

This module defines a Protocol for generic tracers and a concrete
implementation previously known as ``InferredGalaxyZDist``. The concrete
class has been renamed to ``TomographicBin``; an alias ``InferredGalaxyZDist``
is provided for backwards compatibility.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from firecrown.metadata_types._measurements import ALL_MEASUREMENT_TYPES, Measurement
from firecrown.metadata_types._measurements import CMB
from firecrown.metadata_types._utils import TypeSource
from firecrown.utils import YAMLSerializable


@runtime_checkable
class Tracer(Protocol):
    """Protocol describing the minimal tracer interface used across Firecrown.

    Implementations must provide these read-only attributes so code can operate on
    different kinds of tracers (not only galaxy redshift distributions).

    The attributes are declared as read-only properties to remain compatible with
    frozen dataclasses such as ``TomographicBin``.

    :var bin_name: string identifier for the tracer/bin
    :var measurements: set of Measurement values supported by the tracer
    :var type_source: TypeSource describing the source of the tracer
    """

    @property
    def bin_name(self) -> str:  # pragma: no cover - Protocol stub
        ...

    @property
    def measurements(self) -> set[Measurement]:  # pragma: no cover - Protocol stub
        ...

    @property
    def type_source(self) -> TypeSource:  # pragma: no cover - Protocol stub
        ...


@dataclass(frozen=True, kw_only=True)
class TomographicBin(YAMLSerializable):
    """Concrete tomographic bin holding redshift distribution arrays.

    This class is the renamed replacement for the legacy
    ``InferredGalaxyZDist`` type. It is intentionally frozen and lightweight.
    """

    bin_name: str
    z: np.ndarray
    dndz: np.ndarray
    measurements: set[Measurement]
    type_source: TypeSource = TypeSource.DEFAULT

    def __post_init__(self) -> None:
        """Validate the redshift resolution data.

        - Make sure the z and dndz arrays have the same shape;
        - The measurement entries must be members of Measurement enum;
        - The bin_name should not be empty.
        """
        if self.z.shape != self.dndz.shape:
            raise ValueError("The z and dndz arrays should have the same shape.")

        for measurement in self.measurements:
            if not isinstance(measurement, ALL_MEASUREMENT_TYPES):
                raise ValueError("The measurement should be a Measurement.")

        if self.bin_name == "":
            raise ValueError("The bin_name should not be empty.")

    def __eq__(self, other: object) -> bool:
        """Equality test for TomographicBin.

        Two TomographicBin objects are equal if they have equal bin_name, z,
        dndz, and measurements.

        If ``other`` is another concrete ``TomographicBin`` instance we perform
        the full array and set comparisons. If ``other`` implements the
        :class:`Tracer` protocol but is not a ``TomographicBin``, return False
        (they are distinct implementations). For unrelated types return
        ``NotImplemented`` so Python can try the reflected operation.
        """
        if isinstance(other, TomographicBin):
            return (
                self.bin_name == other.bin_name
                and np.array_equal(self.z, other.z)
                and np.array_equal(self.dndz, other.dndz)
                and self.measurements == other.measurements
            )

        # Another object that satisfies the Tracer protocol: intentionally
        # considered not equal to this concrete TomographicBin implementation.
        if isinstance(other, Tracer):
            return False

        # Allow Python to attempt other.__eq__(self)
        return NotImplemented

    @property
    def measurement_list(self) -> list[Measurement]:
        """Get the measurements as a sorted list."""
        return sorted(self.measurements)
        # Exact same concrete type: compare contents


# Backwards compatibility: keep the old name available
InferredGalaxyZDist = TomographicBin


@dataclass(frozen=True, kw_only=True)
class CMBLensing(YAMLSerializable):
    """Tracer type describing CMB lensing.

    This minimal tracer exposes the same public interface as other tracers
    (``bin_name``, ``measurements``, ``type_source``) and additionally
    stores the redshift of the last-scattering surface as ``z_lss``.

    The only measurement supported by this tracer by default is
    :data:`firecrown.metadata_types._measurements.CMB.CONVERGENCE`.
    """

    bin_name: str
    z_lss: float
    measurements: set[Measurement] = field(default_factory=lambda: {CMB.CONVERGENCE})
    type_source: TypeSource = TypeSource.DEFAULT

    def __post_init__(self) -> None:
        """Basic validation for the CMB lensing tracer.

        - Ensure bin_name is not empty;
        - Ensure z_lss is a finite non-negative number; and
        - Ensure measurements contain valid Measurement members.
        """
        if self.bin_name == "":
            raise ValueError("The bin_name should not be empty.")

        if not isinstance(self.z_lss, (int, float)) or not np.isfinite(self.z_lss):
            raise ValueError("z_lss must be a finite float value.")

        if self.z_lss < 0:
            raise ValueError("z_lss must be non-negative.")

        for measurement in self.measurements:
            if not isinstance(measurement, ALL_MEASUREMENT_TYPES):
                raise ValueError("The measurement should be a Measurement.")

    def __eq__(self, other: object) -> bool:
        """Equality semantics for CMBLensing.

        Two CMBLensing instances are equal if they have the same bin_name,
        z_lss, measurements and type_source. If ``other`` implements the
        Tracer protocol but is not a CMBLensing instance, we return False.
        For unrelated types, return NotImplemented.
        """
        if isinstance(other, CMBLensing):
            return (
                self.bin_name == other.bin_name
                and self.z_lss == other.z_lss
                and self.measurements == other.measurements
                and self.type_source == other.type_source
            )

        if isinstance(other, Tracer):
            return False

        return NotImplemented
