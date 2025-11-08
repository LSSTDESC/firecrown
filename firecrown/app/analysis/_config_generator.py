"""Framework-specific configuration generators.

Provides a stateful strategy pattern for generating configuration files
through phased construction: add components incrementally, then write.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

from pathlib import Path

from ._types import Frameworks, ConfigGenerator
from ._cosmosis import CosmosisConfigGenerator
from ._numcosmo import NumCosmoConfigGenerator
from ._cobaya import CobayaConfigGenerator


def get_generator(
    framework: Frameworks, output_path: Path, prefix: str, use_absolute_path: bool
) -> ConfigGenerator:
    """Factory function to create framework-specific configuration generator.

    :param framework: Target framework (cosmosis, cobaya, numcosmo)
    :param output_path: Directory where configuration files will be written
    :param prefix: Prefix for generated filenames
    :param use_absolute_path: Use absolute paths in configuration files
    :return: Initialized configuration generator
    :raises ValueError: If framework is not supported
    """
    match framework:
        case Frameworks.COSMOSIS:

            return CosmosisConfigGenerator(
                output_path=output_path,
                prefix=prefix,
                use_absolute_path=use_absolute_path,
            )
        case Frameworks.COBAYA:
            return CobayaConfigGenerator(
                output_path=output_path,
                prefix=prefix,
                use_absolute_path=use_absolute_path,
            )
        case Frameworks.NUMCOSMO:
            return NumCosmoConfigGenerator(
                output_path=output_path,
                prefix=prefix,
                use_absolute_path=use_absolute_path,
            )
        case _:
            raise ValueError(f"Unsupported framework: {framework}")
