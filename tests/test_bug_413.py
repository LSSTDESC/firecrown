"""Test for TwoPoint calculate_pk."""

from typing import Tuple
import pyccl
import sacc

from firecrown.likelihood.number_counts import (
    NumberCounts,
    PTNonLinearBiasSystematic,
)
from firecrown.likelihood.two_point import TwoPoint
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools


def make_twopoint_with_optional_systematics(
    sacc_data: sacc.Sacc,
    first_source_has_systematic: bool,
    second_source_has_systematic: bool,
) -> Tuple[NumberCounts, NumberCounts, TwoPoint, ModelingTools]:
    """Create a TwoPoint object with optional PT systematics."""

    # Make first source
    if first_source_has_systematic:
        systematic = PTNonLinearBiasSystematic(sacc_tracer="lens0")
        a = NumberCounts(sacc_tracer="lens0", systematics=[systematic])
    else:
        a = NumberCounts(sacc_tracer="lens0", systematics=None)

    # Make second source
    if second_source_has_systematic:
        systematic = PTNonLinearBiasSystematic(sacc_tracer="lens1")
        b = NumberCounts(sacc_tracer="lens1", systematics=[systematic])
    else:
        b = NumberCounts(sacc_tracer="lens1", systematics=None)

    statistic = TwoPoint(
        "galaxy_density_xi",
        a,
        b,
        ell_for_xi={"minimum": 2, "midpoint": 6, "n_log": 180},
    )
    statistic.read(sacc_data)
    # Note the two sources have identical parameters.
    param_map = ParamsMap(
        {
            "lens0_bias": 1.1,
            "lens0_b_2": 1.05,
            "lens0_b_s": 0.99,
            "lens1_bias": 1.1,
            "lens1_b_2": 1.05,
            "lens1_b_s": 0.99,
        }
    )
    statistic.update(param_map)
    tools = ModelingTools(
        pt_calculator=pyccl.nl_pt.EulerianPTCalculator(
            with_NC=True,
            with_IA=True,
            log10k_min=-4,
            log10k_max=2,
            nk_per_decade=20,
        )
    )
    tools.update(param_map)
    tools.prepare(pyccl.CosmologyVanillaLCDM())
    _ = a.get_tracers(tools)
    _ = b.get_tracers(tools)
    assert a.tracers[0].has_pt == first_source_has_systematic
    assert b.tracers[0].has_pt == second_source_has_systematic
    return a, b, statistic, tools


def test_calculate_pk(sacc_galaxy_xis_lens0_lens1):
    # Set up a TwoPoint object in which the first source has a PT systematic
    # and the second source doe not have a PT systematic.

    sacc_data, _, _, _ = sacc_galaxy_xis_lens0_lens1

    a, b, statistic, tools = make_twopoint_with_optional_systematics(
        sacc_data, True, False
    )

    # Now we can actually test the thing.
    spectrum_true_false = statistic.calculate_pk(
        "this has not been done before", tools, a.tracers[0], b.tracers[0]
    )
    assert isinstance(spectrum_true_false, pyccl.Pk2D)

    a, b, statistic, tools = make_twopoint_with_optional_systematics(
        sacc_data, False, True
    )
    spectrum_false_true = statistic.calculate_pk(
        "neither has this", tools, a.tracers[0], b.tracers[0]
    )
    assert isinstance(spectrum_false_true, pyccl.Pk2D)
    assert spectrum_true_false == spectrum_false_true
