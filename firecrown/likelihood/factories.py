"""Factory functions for creating likelihoods from SACC files.

This module provides factory functions to create likelihood objects by combining a SACC
file and a set of statistic factories. Users can define their own custom statistic
factories for advanced use cases or rely on the generic factory functions provided here
for simpler scenarios.

For straightforward contexts where all data in the SACC file is utilized, the generic
factories simplify the process. The user only needs to supply the SACC file and specify
which statistic factories to use, and the likelihood factory will handle the creation of
the likelihood object, assembling the necessary components automatically.

These functions are particularly useful when the full set of statistics present in a
SACC file is being used without the need for complex customization.
"""

from pathlib import Path

import sacc
import yaml
from pydantic import BaseModel
from typing_extensions import assert_never

from firecrown.ccl_factory import CCLFactory
from firecrown.data_functions import (
    TwoPointBinFilterCollection,
    check_two_point_consistence_harmonic,
    check_two_point_consistence_real,
    extract_all_harmonic_data,
    extract_all_real_data,
)
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.likelihood.two_point import TwoPointFactory
from firecrown.metadata_types import TwoPointCorrelationSpace
from firecrown.modeling_tools import ModelingTools


def load_sacc_data(filepath: str | Path) -> sacc.Sacc:
    """Load SACC data from a file, auto-detecting the format.

    Attempts to load the file first as HDF5, then as FITS if HDF5 fails.
    This allows the function to work with both modern HDF5-based SACC files
    and legacy FITS-based SACC files.

    :param filepath: Path to the SACC data file (str or Path object)
    :return: Loaded SACC data object
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the file cannot be read as either HDF5 or FITS SACC data
    """
    # Convert to Path object for consistent handling
    file_path = Path(filepath) if isinstance(filepath, str) else filepath

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"SACC file not found: {file_path}")

    # Try HDF5 first (modern format)
    hdf5_error = None
    try:
        return sacc.Sacc.load_hdf5(str(file_path))
    except OSError as e:
        hdf5_error = e

    # If HDF5 failed, try FITS (legacy format)
    fits_error = None
    try:
        return sacc.Sacc.load_fits(str(file_path))
    except OSError as e:
        fits_error = e

    # Both formats failed - provide helpful error message
    raise ValueError(
        f"Failed to load SACC data from file: {file_path}\n"
        f"The file could not be read as either HDF5 or FITS format.\n"
        f"HDF5 error: {hdf5_error}\n"
        f"FITS error: {fits_error}"
    )


class DataSourceSacc(BaseModel):
    """Model for the data source in a likelihood configuration."""

    sacc_data_file: str
    filters: TwoPointBinFilterCollection | None = None
    _path: Path | None = None

    def set_path(self, path: Path) -> None:
        """Set the path for the data source."""
        self._path = path

    def get_filepath(self) -> Path:
        """Return the filename of the data source.

        Raises a FileNotFoundError if the file does not exist.
        :return: The filename
        """
        sacc_data_path = Path(self.sacc_data_file)
        # If sacc_data_file is absolute, use it directly
        if sacc_data_path.is_absolute() and sacc_data_path.exists():
            return Path(self.sacc_data_file)
        # If path is set, use it to find the file
        if self._path is not None:
            full_sacc_data_path = self._path / sacc_data_path
            if full_sacc_data_path.exists():
                return full_sacc_data_path
        # If path is not set, use the current directory
        elif sacc_data_path.exists():
            return sacc_data_path
        # If the file does not exist, raise an error
        raise FileNotFoundError(f"File {sacc_data_path} does not exist")

    def get_sacc_data(self) -> sacc.Sacc:
        """Load the SACC data file.

        Uses automatic format detection to load both HDF5 and FITS files.
        """
        filename = self.get_filepath()
        return load_sacc_data(filename)


def ensure_path(file: str | Path) -> Path:
    """Ensure the file path is a Path object."""
    match file:
        case str():
            return Path(file)
        case Path():
            return file
        case _ as unreachable:
            assert_never(unreachable)


class TwoPointExperiment(BaseModel):
    """Model for the two-point experiment in a likelihood configuration."""

    two_point_factory: TwoPointFactory
    data_source: DataSourceSacc
    ccl_factory: CCLFactory | None = None

    def model_post_init(self, _, /) -> None:
        """Initialize the TwoPointExperiment object."""
        if self.ccl_factory is None:
            self.ccl_factory = CCLFactory()

    @classmethod
    def load_from_yaml(cls, file: str | Path) -> "TwoPointExperiment":
        """Load a TwoPointExperiment object from a YAML file."""
        filepath = ensure_path(file)

        with open(filepath, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            tpe = cls.model_validate(config, strict=True)

        # Record the file directory
        tpe.data_source.set_path(filepath.parent)
        return tpe

    def make_likelihood(self) -> Likelihood:
        """Create a likelihood object for two-point statistics from a SACC file."""
        # Load the SACC file
        sacc_data = self.data_source.get_sacc_data()

        likelihood: None | Likelihood = None
        match self.two_point_factory.correlation_space:
            case TwoPointCorrelationSpace.REAL:
                likelihood = _build_two_point_likelihood_real(
                    sacc_data, self.two_point_factory, filters=self.data_source.filters
                )
            case TwoPointCorrelationSpace.HARMONIC:
                likelihood = _build_two_point_likelihood_harmonic(
                    sacc_data, self.two_point_factory, filters=self.data_source.filters
                )
            case _ as unreachable:
                assert_never(unreachable)
        assert likelihood is not None
        return likelihood


def build_two_point_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """Build a likelihood object for two-point statistics from a SACC file.

    This function creates a likelihood object for two-point statistics using a SACC file
    and a set of statistic factories. The user must provide the SACC file and specify
    which statistic factories to use. The likelihood object is created by combining the
    SACC file with the specified statistic factories.

    :param build_parameters: A NamedParameters object containing the following
        parameters:
        - sacc_file: The SACC file containing the data.
        - statistic_factories: A YAML file containing the statistic factories to use.
    """
    likelihood_config_file = build_parameters.get_string("likelihood_config")
    exp = TwoPointExperiment.load_from_yaml(likelihood_config_file)
    modeling_tools = ModelingTools(ccl_factory=exp.ccl_factory)

    likelihood = exp.make_likelihood()

    return likelihood, modeling_tools


def _build_two_point_likelihood_harmonic(
    sacc_data: sacc.Sacc,
    two_point_factory: TwoPointFactory,
    filters: TwoPointBinFilterCollection | None = None,
):
    """Build a likelihood object for two-point statistics in harmonic space.

    This function creates a likelihood object for two-point statistics in harmonic space
    using a SACC file and a set of statistic factories. The user must provide the SACC
    file and specify which statistic factories to use. The likelihood object is created
    by combining the SACC file with the specified statistic factories.

    :param sacc_data: The SACC file containing the data.
    :param wl_factory: The weak lensing statistic factory.
    :param nc_factory: The number counts statistic factory.

    :return: A likelihood object for two-point statistics in harmonic space.
    """
    tpms = extract_all_harmonic_data(sacc_data)
    if len(tpms) == 0:
        raise ValueError(
            "No two-point measurements in harmonic space found in the SACC file."
        )
    check_two_point_consistence_harmonic(tpms)
    if filters is not None:
        tpms = filters(tpms)

    two_points = two_point_factory.from_measurement(tpms)
    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood


def _build_two_point_likelihood_real(
    sacc_data: sacc.Sacc,
    two_point_factory: TwoPointFactory,
    filters: TwoPointBinFilterCollection | None = None,
):
    """Build a likelihood object for two-point statistics in real space.

    This function creates a likelihood object for two-point statistics in real space
    using a SACC file and a set of statistic factories. The user must provide the SACC
    file and specify which statistic factories to use. The likelihood object is created
    by combining the SACC file with the specified statistic factories.

    :param sacc_data: The SACC file containing the data.
    :param wl_factory: The weak lensing statistic factory.
    :param nc_factory: The number counts statistic factory.

    :return: A likelihood object for two-point statistics in real space.
    """
    tpms = extract_all_real_data(sacc_data)
    if len(tpms) == 0:
        raise ValueError(
            "No two-point measurements in real space found in the SACC file."
        )
    check_two_point_consistence_real(tpms)
    if filters is not None:
        tpms = filters(tpms)

    two_points = two_point_factory.from_measurement(tpms)
    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood
