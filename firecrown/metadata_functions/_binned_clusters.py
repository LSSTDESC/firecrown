"""Extraction of binned cluster metadata from SACC objects."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt
import sacc
from sacc.tracers import BinRichnessTracer, BinZTracer, SurveyTracer

from firecrown.metadata_types import (
    BinnedClusterMetadata,
    BinnedClusterObservable,
    ClusterBin,
    ClusterSurvey,
)

_TracerT = TypeVar("_TracerT", bound=sacc.BaseTracer)


def _get_tracer(
    sacc_data: sacc.Sacc, name: str, tracer_class: type[_TracerT]
) -> _TracerT:
    """Return the tracer called name, checking that it is a tracer_class."""
    tracer = sacc_data.get_tracer(name)
    if not isinstance(tracer, tracer_class):
        raise ValueError(
            f"The SACC tracer {name} should be a {tracer_class.__name__}, "
            f"found {type(tracer).__name__}."
        )
    return tracer


def _make_bin(sacc_data: sacc.Sacc, z_name: str, proxy_name: str) -> ClusterBin:
    """Return the ClusterBin of a BinZTracer and a BinRichnessTracer."""
    z_tracer = _get_tracer(sacc_data, z_name, BinZTracer)
    proxy_tracer = _get_tracer(sacc_data, proxy_name, BinRichnessTracer)
    return ClusterBin(
        z_tracer=z_name,
        z_edges=(float(z_tracer.lower), float(z_tracer.upper)),
        proxy_tracer=proxy_name,
        proxy_edges=(float(proxy_tracer.lower), float(proxy_tracer.upper)),
    )


def extract_all_binned_cluster_metadata_indices(
    sacc_data: sacc.Sacc,
) -> list[tuple[BinnedClusterMetadata, npt.NDArray[np.int64]]]:
    """Return the metadata and SACC indices of every binned cluster observable.

    Each data point of a BinnedClusterObservable data type must have the tracers
    (SurveyTracer, BinZTracer, BinRichnessTracer). The data points are grouped by
    observable and survey, keeping their SACC order; each group gives one
    BinnedClusterMetadata.

    :param sacc_data: The SACC object.
    :returns: One (metadata, indices) pair per observable and survey.
    """
    result: list[tuple[BinnedClusterMetadata, npt.NDArray[np.int64]]] = []
    for observable in BinnedClusterObservable:
        groups: dict[str, list[int]] = {}
        for index in sacc_data.indices(data_type=observable.value):
            tracers = sacc_data.data[index].tracers
            if len(tracers) != 3:
                raise ValueError(
                    f"The {observable.value} data point {index} should have 3 "
                    f"tracers (survey, z bin, richness bin), found {tracers}."
                )
            groups.setdefault(tracers[0], []).append(int(index))

        for survey_name, indices in groups.items():
            survey_tracer = _get_tracer(sacc_data, survey_name, SurveyTracer)
            survey = ClusterSurvey(
                name=survey_name, sky_area=float(survey_tracer.sky_area)
            )
            bins = tuple(
                _make_bin(sacc_data, *sacc_data.data[i].tracers[1:]) for i in indices
            )
            metadata = BinnedClusterMetadata(
                survey=survey, observable=observable, bins=bins
            )
            result.append((metadata, np.array(indices, dtype=np.int64)))
    return result
