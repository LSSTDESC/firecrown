"""Data models for likelihood factory configuration."""

from collections.abc import Sequence
from pathlib import Path

import sacc
import yaml
from pydantic import BaseModel
from typing_extensions import assert_never

from firecrown.data_functions import (
    TwoPointBinFilterCollection,
    check_two_point_consistence_harmonic,
    check_two_point_consistence_real,
    extract_all_harmonic_data,
    extract_all_real_data,
)

# The following imports from private modules are needed
# to avoid circular dependencies.
from firecrown.likelihood._gaussian import ConstGaussian
from firecrown.likelihood._two_point import TwoPointFactory
from firecrown.likelihood.clusters._binned_cluster_factory import (
    BinnedClusterFactory,
)
from firecrown.likelihood_base import Likelihood, Statistic
from firecrown.metadata_types import TwoPointCorrelationSpace
from firecrown.modeling_tools import CCLFactory
from firecrown.utils import ensure_path, load_sacc_data


def _add_cluster_statistics(
    statistics: Sequence[Statistic],
    sacc_data: sacc.Sacc,
    cluster_factory: BinnedClusterFactory | None,
) -> Sequence[Statistic]:
    """Append the cluster statistics requested for a likelihood.

    :param statistics: The two-point statistics already built.
    :param sacc_data: The SACC file containing the data.
    :param cluster_factory: The binned cluster factory, or None.

    :returns: The statistics, followed by the cluster statistics.
    """
    if cluster_factory is None:
        return statistics
    return [*statistics, *cluster_factory.from_sacc(sacc_data)]


def _build_two_point_likelihood_harmonic(
    sacc_data: sacc.Sacc,
    two_point_factory: TwoPointFactory,
    filters: TwoPointBinFilterCollection | None = None,
    normalize: bool = True,
    cluster_factory: BinnedClusterFactory | None = None,
) -> Likelihood:
    """Build a likelihood object for two-point statistics in harmonic space.

    This function creates a likelihood object for two-point statistics in harmonic
    space using a SACC file and a set of statistic factories. The user must provide the
    SACC file and specify which statistic factories to use. The likelihood object is
    created by combining the SACC file with the specified statistic factories.

    :param sacc_data: The SACC file containing the data.
    :param two_point_factory: The two-point statistic factory.
    :param filters: Optional filters to apply to the two-point measurements.
    :param normalize: If True, normalize the window function weights to sum to 1.
    :param cluster_factory: Optional binned cluster factory whose statistics are
        added to the same likelihood.

    :returns: A likelihood object for two-point statistics in harmonic space.
    """
    tpms = extract_all_harmonic_data(sacc_data, normalize=normalize)
    if len(tpms) == 0:
        raise ValueError(
            "No two-point measurements in harmonic space found in the SACC file."
        )
    check_two_point_consistence_harmonic(tpms)
    if filters is not None:
        tpms = filters(tpms)

    statistics = _add_cluster_statistics(
        two_point_factory.from_measurement(tpms), sacc_data, cluster_factory
    )
    assert sacc_data.covariance is not None
    likelihood = ConstGaussian.create_ready(statistics, sacc_data.covariance.dense)

    return likelihood


def _build_two_point_likelihood_real(
    sacc_data: sacc.Sacc,
    two_point_factory: TwoPointFactory,
    filters: TwoPointBinFilterCollection | None = None,
    cluster_factory: BinnedClusterFactory | None = None,
) -> Likelihood:
    """Build a likelihood object for two-point statistics in real space.

    This function creates a likelihood object for two-point statistics in real space
    using a SACC file and a set of statistic factories. The user must provide the SACC
    file and specify which statistic factories to use. The likelihood object is created
    by combining the SACC file with the specified statistic factories.

    :param sacc_data: The SACC file containing the data.
    :param two_point_factory: The two-point statistic factory.
    :param filters: Optional filters to apply to the two-point measurements.
    :param cluster_factory: Optional binned cluster factory whose statistics are
        added to the same likelihood.

    :returns: A likelihood object for two-point statistics in real space.
    """
    tpms = extract_all_real_data(sacc_data)
    if len(tpms) == 0:
        raise ValueError(
            "No two-point measurements in real space found in the SACC file."
        )
    check_two_point_consistence_real(tpms)
    if filters is not None:
        tpms = filters(tpms)

    statistics = _add_cluster_statistics(
        two_point_factory.from_measurement(tpms), sacc_data, cluster_factory
    )
    assert sacc_data.covariance is not None
    likelihood = ConstGaussian.create_ready(statistics, sacc_data.covariance.dense)

    return likelihood


def _build_cluster_likelihood(
    sacc_data: sacc.Sacc, cluster_factory: BinnedClusterFactory
) -> Likelihood:
    """Build a likelihood object for binned cluster statistics.

    :param sacc_data: The SACC file containing the data.
    :param cluster_factory: The binned cluster factory of all cluster samples.

    :returns: A likelihood object for the cluster statistics.
    """
    statistics = cluster_factory.from_sacc(sacc_data)
    if sacc_data.covariance is None:
        raise ValueError(
            "The SACC data has no covariance; one is required to build the likelihood."
        )
    return ConstGaussian.create_ready(statistics, sacc_data.covariance.dense)


class DataSourceSacc(BaseModel):
    """Model for the data source in a likelihood configuration."""

    sacc_data_file: str
    filters: TwoPointBinFilterCollection | None = None
    normalize_window: bool = True
    _path: Path | None = None

    def set_path(self, path: Path) -> None:
        """Set the path for the data source."""
        self._path = path

    def get_filepath(self) -> Path:
        """Return the filename of the data source.

        Raises a FileNotFoundError if the file does not exist.
        :returns: The filename
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
    """Model for the two-point experiment in a likelihood configuration.

    A binned cluster factory may be added; its statistics enter the same
    Gaussian likelihood, using the covariance stored in the SACC file.
    """

    two_point_factory: TwoPointFactory
    data_source: DataSourceSacc
    ccl_factory: CCLFactory | None = None
    cluster_factory: BinnedClusterFactory | None = None

    def model_post_init(self, _: object, /) -> None:
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
                    sacc_data,
                    self.two_point_factory,
                    filters=self.data_source.filters,
                    cluster_factory=self.cluster_factory,
                )
            case TwoPointCorrelationSpace.HARMONIC:
                likelihood = _build_two_point_likelihood_harmonic(
                    sacc_data,
                    self.two_point_factory,
                    filters=self.data_source.filters,
                    normalize=self.data_source.normalize_window,
                    cluster_factory=self.cluster_factory,
                )
            case _ as unreachable:
                assert_never(unreachable)
        assert likelihood is not None
        return likelihood


class ClusterExperiment(BaseModel):
    """Model for a binned galaxy cluster experiment in a likelihood configuration.

    The likelihood is a :class:`ConstGaussian` over the statistics of the
    cluster factory, using the covariance stored in the SACC file.
    """

    cluster_factory: BinnedClusterFactory
    data_source: DataSourceSacc
    ccl_factory: CCLFactory | None = None

    def model_post_init(self, _: object, /) -> None:
        """Initialize the ClusterExperiment object."""
        if self.ccl_factory is None:
            self.ccl_factory = CCLFactory()

    @classmethod
    def load_from_yaml(cls, file: str | Path) -> "ClusterExperiment":
        """Load a ClusterExperiment object from a YAML file."""
        filepath = ensure_path(file)

        with open(filepath, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            experiment = cls.model_validate(config, strict=True)

        # Record the file directory
        experiment.data_source.set_path(filepath.parent)
        return experiment

    def make_likelihood(self) -> Likelihood:
        """Create a likelihood object for binned cluster statistics from a SACC file."""
        if self.data_source.filters is not None:
            raise ValueError(
                "Two-point filters cannot be applied to a cluster experiment."
            )
        sacc_data = self.data_source.get_sacc_data()
        return _build_cluster_likelihood(sacc_data, self.cluster_factory)
