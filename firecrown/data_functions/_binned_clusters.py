"""Extraction of binned cluster data from SACC objects."""

import sacc

from firecrown.data_functions._utils import cov_hash
from firecrown.data_types import BinnedClusterMeasurement
from firecrown.metadata_functions import extract_all_binned_cluster_metadata_indices


def extract_all_binned_cluster_data(
    sacc_data: sacc.Sacc,
) -> list[BinnedClusterMeasurement]:
    """Return one BinnedClusterMeasurement per binned cluster observable and survey.

    :param sacc_data: The SACC object; it must contain a covariance matrix.
    :returns: The measurements, in the order of
        extract_all_binned_cluster_metadata_indices.
    """
    if sacc_data.covariance is None:
        raise ValueError("The SACC object should contain a covariance matrix.")
    covariance_name = cov_hash(sacc_data)
    return [
        BinnedClusterMeasurement(
            data=sacc_data.mean[indices],
            indices=indices,
            covariance_name=covariance_name,
            metadata=metadata,
        )
        for metadata, indices in extract_all_binned_cluster_metadata_indices(sacc_data)
    ]
