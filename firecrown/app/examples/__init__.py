"""Example generators for Firecrown analyses.

Provides ready-to-use example generators for different cosmological analyses.
Each example creates complete analysis setups including data files, factory files,
and framework-specific configurations.
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

Maps CLI-friendly names to generator classes. Add new examples here
to make them discoverable via the command-line interface.
"""
