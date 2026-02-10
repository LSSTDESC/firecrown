"""Tests for the NumberCountsSource base class."""

import firecrown.likelihood.number_counts as nc
import firecrown.metadata_types as mt
import firecrown.modeling_tools as mtools
from firecrown import updatable


def test_get_derived_parameters(
    harmonic_bin_1: mt.TomographicBin,
    tools_with_vanilla_cosmology: mtools.ModelingTools,
):
    ncs = nc.NumberCounts.create_ready(harmonic_bin_1, derived_scale=True)
    ncs.update(updatable.ParamsMap({"bin_1_bias": 1.0}))
    ncs.create_tracers(tools_with_vanilla_cosmology)
    params = ncs.get_derived_parameters()
    assert params is not None
    assert len(params) == 1
    for param in params:
        a, b, c = param
        assert a == "TwoPoint"
        assert b == "NumberCountsScale_bin_1"
        assert c == 1.0
