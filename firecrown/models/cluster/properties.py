"""Module containing classes relevant to defining cluster properties."""

from enum import Flag, auto


class ClusterProperty(Flag):
    """Flag representing cluster observables.

    This flag containing the possible cluster properties we can make a theoretical
    prediction for.
    """

    NONE = 0
    COUNTS = auto()
    MASS = auto()
    REDSHIFT = auto()
    SHEAR = auto()
    # SHEAR_APPROX = auto()
    DELTASIGMA = auto()
