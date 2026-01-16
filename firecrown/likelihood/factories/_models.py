"""Data models for likelihood factory configuration."""

from pathlib import Path

import sacc
import yaml
from pydantic import BaseModel
from typing_extensions import assert_never

from firecrown.modeling_tools import CCLFactory
from firecrown.data_functions import (
    TwoPointBinFilterCollection,
    check_two_point_consistence_harmonic,
    check_two_point_consistence_real,
    extract_all_harmonic_data,
    extract_all_real_data,
)
from firecrown.likelihood._gaussian import ConstGaussian
from firecrown.likelihood._base import Likelihood
from firecrown.likelihood._two_point import TwoPointFactory
from firecrown.metadata_types import TwoPointCorrelationSpace
from firecrown.likelihood.factories._sacc_utils import ensure_path, load_sacc_data


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
    assert sacc_data.covariance is not None
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
    assert sacc_data.covariance is not None
    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood


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
