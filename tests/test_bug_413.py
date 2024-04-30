"""Test for TwoPoint calculate_pk."""

import pyccl

from firecrown.likelihood.gauss_family.statistic.source.number_counts import (
    NumberCounts,
    PTNonLinearBiasSystematic,
)
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools


def test_calculate_pk(sacc_galaxy_xis_lens0_lens1):
    # Set up a TwoPoint object in which the first source has a PT systematic
    # and the second source doe not have a PT systematic.

    sacc_data, _, _, _ = sacc_galaxy_xis_lens0_lens1

    a = NumberCounts(sacc_tracer="lens1", systematics=None)

    systematic = PTNonLinearBiasSystematic(sacc_tracer="lens0")
    b = NumberCounts(sacc_tracer="lens0", systematics=[systematic])

    statistic = TwoPoint(
        "galaxy_density_xi",
        b,
        a,
        ell_for_xi={"minimum": 2, "midpoint": 6, "n_log": 180},
    )
    statistic.read(sacc_data)

    param_map = ParamsMap(
        {"lens1_bias": 1.0, "lens0_bias": 1.1, "lens0_b_2": 1.05, "lens0_b_s": 0.99}
    )
    statistic.update(param_map)

    tools = ModelingTools()
    tools.update(param_map)
    tools.prepare(pyccl.CosmologyVanillaLCDM())

    _ = a.get_tracers(tools)
    _ = b.get_tracers(tools)

    assert not a.tracers[0].has_pt
    assert b.tracers[0].has_pt

    # Now we can actually test the thing.
    spectrum = statistic.calculate_pk(
        "this has not been done before", tools, b.tracers[0], a.tracers[0]
    )
    assert isinstance(spectrum, pyccl.Pk2D)
