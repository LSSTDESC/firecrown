"""Example configurations and generators for Firecrown analyses.

This module provides a framework for generating example configurations
for different types of cosmological analyses supported by Firecrown.
Each example includes the necessary data files and configuration files
to run a complete analysis.
"""

from ._base_example import Example
from ._cosmic_shear import ExampleCosmicShear
from ._sn_srd import ExampleSupernovaSRD
from ._des_y1_3x2pt import ExampleDESY13x2pt


EXAMPLES_LIST: dict[str, type[Example]] = {
    "cosmic_shear": ExampleCosmicShear,
    "sn_srd": ExampleSupernovaSRD,
    "des_y1_3x2pt": ExampleDESY13x2pt,
}
"""Registry of available example generators.

Maps example names to their corresponding generator classes.
New examples should be added here to make them discoverable.
"""
