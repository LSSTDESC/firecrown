"""Framework-specific configuration generators.

Provides a stateful strategy pattern for generating configuration files
through phased construction: add components incrementally, then write.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

from pathlib import Path

from firecrown.ccl_factory import PoweSpecAmplitudeParameter
from ._types import Frameworks, ConfigGenerator, FrameworkCosmology
from ._cosmosis import CosmosisConfigGenerator
from ._numcosmo import NumCosmoConfigGenerator
from ._cobaya import CobayaConfigGenerator


def get_generator(
    framework: Frameworks,
    output_path: Path,
    prefix: str,
    use_absolute_path: bool,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
    amplitude_parameter: PoweSpecAmplitudeParameter = PoweSpecAmplitudeParameter.SIGMA8,
) -> ConfigGenerator:
    """Factory function to create framework-specific configuration generator.

    :param framework: Target framework (cosmosis, cobaya, numcosmo)
    :param output_path: Directory where configuration files will be written
    :param prefix: Prefix for generated filenames
    :param use_absolute_path: Use absolute paths in configuration files
    :param require_cosmology: Include cosmology computation in configuration
    :return: Initialized configuration generator
    :raises ValueError: If framework is not supported
    """
    match framework:
        case Frameworks.COSMOSIS:

            return CosmosisConfigGenerator(
                output_path=output_path,
                prefix=prefix,
                use_absolute_path=use_absolute_path,
                required_cosmology=required_cosmology,
                amplitude_parameter=amplitude_parameter,
            )
        case Frameworks.COBAYA:
            return CobayaConfigGenerator(
                output_path=output_path,
                prefix=prefix,
                use_absolute_path=use_absolute_path,
                required_cosmology=required_cosmology,
                amplitude_parameter=amplitude_parameter,
            )
        case Frameworks.NUMCOSMO:
            return NumCosmoConfigGenerator(
                output_path=output_path,
                prefix=prefix,
                use_absolute_path=use_absolute_path,
                required_cosmology=required_cosmology,
                amplitude_parameter=amplitude_parameter,
            )
        case _:
            raise ValueError(f"Unsupported framework: {framework}")
