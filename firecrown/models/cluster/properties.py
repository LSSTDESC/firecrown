from enum import Flag, auto


class ClusterProperty(Flag):
    NONE = 0
    COUNTS = auto()
    MASS = auto()
    REDSHIFT = auto()
    SHEAR = auto()
