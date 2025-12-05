"""Tests for the BinPairSelector classes in firecrown.metadata_types.

This module contains comprehensive tests for all bin pair selector implementations,
including atomic selectors, composite selectors, logical combinators, and serialization.
"""

import re
import pytest
import numpy as np
import yaml

import firecrown.metadata_types as mt
from firecrown.metadata_functions import make_binned_two_point_filtered


def test_pair_selector_register_missing_kind():
    with pytest.raises(
        ValueError, match="MissingBinPairSelector has no default for 'kind'"
    ):

        @mt.register_bin_pair_selector
        class MissingBinPairSelector(mt.BinPairSelector):
            """BinPairSelector with missing kind."""

            def keep(
                self, _zdist: mt.TomographicBinPair, _m: mt.MeasurementPair
            ) -> bool:
                return True

        _ = MissingBinPairSelector(kind="foo")


def test_pair_selector_register_duplicate_kind():
    with pytest.raises(ValueError, match="Duplicate pair selector kind foo"):

        @mt.register_bin_pair_selector
        class Duplicate1BinPairSelector(mt.BinPairSelector):
            """BinPairSelector with duplicate kind."""

            kind: str = "foo"

            def keep(
                self, _zdist: mt.TomographicBinPair, _m: mt.MeasurementPair
            ) -> bool:
                return True

        @mt.register_bin_pair_selector
        class Duplicate2BinPairSelector(mt.BinPairSelector):
            """BinPairSelector with duplicate kind."""

            kind: str = "foo"

            def keep(
                self, _zdist: mt.TomographicBinPair, _m: mt.MeasurementPair
            ) -> bool:
                return True

        _ = Duplicate2BinPairSelector()
        _ = Duplicate1BinPairSelector()


def test_pair_selector_auto(all_harmonic_bins):
    auto_pair_selector = (
        mt.AutoNameBinPairSelector() & mt.AutoMeasurementBinPairSelector()
    )

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, auto_pair_selector
    )
    # AutoBinPairSelector should create all auto-combinations
    # There is two measurements per bin in all_harmonic_bins
    assert len(two_point_xy_combinations) == 2 * len(all_harmonic_bins)
    for two_point_xy in two_point_xy_combinations:
        assert two_point_xy.x == two_point_xy.y
        assert two_point_xy.x_measurement == two_point_xy.y_measurement


def test_pair_selector_auto_source(all_harmonic_bins):
    auto_pair_selector = (
        mt.AutoNameBinPairSelector() & mt.AutoMeasurementBinPairSelector()
    )
    source_pair_selector = mt.SourceBinPairSelector()

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, auto_pair_selector & source_pair_selector
    )
    # AutoBinPairSelector should create all auto-combinations for shear measurements
    assert len(two_point_xy_combinations) == 2
    for two_point_xy in two_point_xy_combinations:
        assert two_point_xy.x == two_point_xy.y
        assert two_point_xy.x_measurement == two_point_xy.y_measurement
        assert two_point_xy.x_measurement in mt.GALAXY_SOURCE_TYPES


def test_pair_selector_auto_lens(all_harmonic_bins):
    auto_pair_selector = (
        mt.AutoNameBinPairSelector() & mt.AutoMeasurementBinPairSelector()
    )
    lens_pair_selector = mt.LensBinPairSelector()

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, auto_pair_selector & lens_pair_selector
    )
    # AutoBinPairSelector should create all auto-combinations for lens measurements
    assert len(two_point_xy_combinations) == 2
    for two_point_xy in two_point_xy_combinations:
        assert two_point_xy.x == two_point_xy.y
        assert two_point_xy.x_measurement == two_point_xy.y_measurement
        assert two_point_xy.x_measurement in mt.GALAXY_LENS_TYPES


def test_pair_selector_auto_source_lens(all_harmonic_bins):
    auto_pair_selector = (
        mt.AutoNameBinPairSelector() & mt.AutoMeasurementBinPairSelector()
    )
    source_pair_selector = mt.SourceBinPairSelector()
    lens_pair_selector = mt.LensBinPairSelector()

    pair_selector = (auto_pair_selector & lens_pair_selector) | (
        auto_pair_selector & source_pair_selector
    )
    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, pair_selector
    )
    # AutoBinPairSelector should create all auto-combinations for lens measurements
    assert len(two_point_xy_combinations) == 4
    for two_point_xy in two_point_xy_combinations:
        assert two_point_xy.x == two_point_xy.y
        assert two_point_xy.x_measurement == two_point_xy.y_measurement


def test_pair_selector_named(all_harmonic_bins):
    named_pair_selector = mt.NamedBinPairSelector(names=[("bin_1", "bin_2")])

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, named_pair_selector
    )
    # NamedBinPairSelector should create all named combinations
    assert len(two_point_xy_combinations) == 3
    for two_point_xy in two_point_xy_combinations:
        assert {two_point_xy.x.bin_name, two_point_xy.y.bin_name} == {"bin_1", "bin_2"}


def test_pair_selector_type_source(all_harmonic_bins):
    type_source_pair_selector = mt.TypeSourceBinPairSelector(
        type_source=mt.TypeSource.DEFAULT
    )

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, type_source_pair_selector
    )
    # TypeSourceBinPairSelector should create all type-source combinations
    assert len(two_point_xy_combinations) == 10

    z1 = mt.InferredGalaxyZDist(
        bin_name="extra_src1",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
        type_source=mt.TypeSource("NewTypeSource"),
    )

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins + [z1], type_source_pair_selector
    )
    assert len(two_point_xy_combinations) == 10
    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins + [z1],
        mt.TypeSourceBinPairSelector(type_source=mt.TypeSource("NewTypeSource")),
    )
    assert len(two_point_xy_combinations) == 1


def test_pair_selector_not_named(all_harmonic_bins):
    named_pair_selector = mt.NamedBinPairSelector(names=[("bin_1", "bin_2")])

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, ~named_pair_selector
    )
    # NamedBinPairSelector should create all named combinations
    assert len(two_point_xy_combinations) == 7
    for two_point_xy in two_point_xy_combinations:
        assert [two_point_xy.x.bin_name, two_point_xy.y.bin_name] != ["bin_1", "bin_2"]


def test_pair_selector_first_neighbor(many_harmonic_bins):
    first_neighbor_pair_selector = mt.FirstNeighborsBinPairSelector()

    z1 = mt.InferredGalaxyZDist(
        bin_name="extra_src_a",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    z2 = mt.InferredGalaxyZDist(
        bin_name="extra_src_b",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    two_point_xy_combinations = make_binned_two_point_filtered(
        many_harmonic_bins + [z1, z2], first_neighbor_pair_selector
    )
    # FirstNeighborBinPairSelector should create all first neighbor combinations
    assert len(two_point_xy_combinations) == 31
    for two_point_xy in two_point_xy_combinations:
        assert re.match(r".*?\d+$", two_point_xy.x.bin_name)
        assert re.match(r".*?\d+$", two_point_xy.y.bin_name)
        index1 = int(re.findall(r"\d+$", two_point_xy.x.bin_name)[0])
        index2 = int(re.findall(r"\d+$", two_point_xy.y.bin_name)[0])
        assert abs(index1 - index2) <= 1


def test_pair_selector_first_neighbor_no_auto(many_harmonic_bins):
    first_neighbor_pair_selector = mt.FirstNeighborsBinPairSelector()
    auto_pair_selector = mt.AutoNameBinPairSelector()
    first_neighbor_no_auto_pair_selector = (
        first_neighbor_pair_selector & ~auto_pair_selector
    )

    two_point_xy_combinations = make_binned_two_point_filtered(
        many_harmonic_bins, first_neighbor_no_auto_pair_selector
    )
    # FirstNeighborBinPairSelector should create all first neighbor combinations
    assert len(two_point_xy_combinations) == 16
    for two_point_xy in two_point_xy_combinations:
        assert re.match(r".*?\d+$", two_point_xy.x.bin_name)
        assert re.match(r".*?\d+$", two_point_xy.y.bin_name)
        index1 = int(re.findall(r"\d+$", two_point_xy.x.bin_name)[0])
        index2 = int(re.findall(r"\d+$", two_point_xy.y.bin_name)[0])
        assert abs(index1 - index2) == 1


def test_pair_selector_auto_name_keep():
    z1 = mt.InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    z2 = mt.InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    rule = mt.AutoNameBinPairSelector()
    assert rule.keep((z1, z1), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    assert not rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))


def test_pair_selector_auto_measurement_keep():
    z1 = mt.InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    z2 = mt.InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    rule = mt.AutoMeasurementBinPairSelector()
    assert rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    assert not rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.SHEAR_E))


def test_pair_selector_named_keep():
    z1 = mt.InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    z2 = mt.InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    rule = mt.NamedBinPairSelector(names=[("bin_1", "bin_2")])
    assert rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))


def test_pair_selector_lens_keep():
    z1 = mt.InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    z2 = mt.InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    rule = mt.LensBinPairSelector()
    assert rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    assert not rule.keep((z1, z2), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))


def test_pair_selector_source_keep():
    z1 = mt.InferredGalaxyZDist(
        bin_name="src1",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    z2 = mt.InferredGalaxyZDist(
        bin_name="src1",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    rule = mt.SourceBinPairSelector()
    assert rule.keep((z1, z2), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))


def test_pair_selector_first_neighbor_mixed():
    z1 = mt.InferredGalaxyZDist(
        bin_name="extra_src_a",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    z2 = mt.InferredGalaxyZDist(
        bin_name="extra_src_b",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    z3 = mt.InferredGalaxyZDist(
        bin_name="extra_src_1",
        z=np.array([0.3]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    rule = mt.FirstNeighborsBinPairSelector()
    assert not rule.keep((z1, z2), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert not rule.keep((z1, z3), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert not rule.keep((z2, z3), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert not rule.keep((z3, z1), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert not rule.keep((z3, z2), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert rule.keep((z3, z3), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))


def test_pair_selector_serialization_and_or():
    pair_selector = (
        mt.AutoNameBinPairSelector()
        & mt.AutoMeasurementBinPairSelector()
        & mt.LensBinPairSelector()
    ) | (
        mt.AutoNameBinPairSelector()
        & mt.AutoMeasurementBinPairSelector()
        & mt.SourceBinPairSelector()
    )

    yaml_str = yaml.dump(
        pair_selector.model_dump(),
        sort_keys=False,
    )

    pair_selector_from_yaml = mt.BinPairSelector.model_validate(
        yaml.safe_load(yaml_str)
    )
    assert pair_selector == pair_selector_from_yaml
    assert pair_selector.model_dump() == pair_selector_from_yaml.model_dump()


def test_pair_selector_serialization_simple_and():
    pair_selector = mt.AutoNameBinPairSelector() & mt.AutoMeasurementBinPairSelector()
    yaml_str = yaml.dump(pair_selector.model_dump(), sort_keys=False)
    pair_selector_from_yaml = mt.BinPairSelector.model_validate(
        yaml.safe_load(yaml_str)
    )
    assert pair_selector == pair_selector_from_yaml
    assert pair_selector.model_dump() == pair_selector_from_yaml.model_dump()


def test_pair_selector_serialization_simple_or():
    pair_selector = mt.AutoNameBinPairSelector() | mt.LensBinPairSelector()
    yaml_str = yaml.dump(pair_selector.model_dump(), sort_keys=False)
    pair_selector_from_yaml = mt.BinPairSelector.model_validate(
        yaml.safe_load(yaml_str)
    )
    assert pair_selector == pair_selector_from_yaml
    assert pair_selector.model_dump() == pair_selector_from_yaml.model_dump()


def test_pair_selector_serialization_nested_and_or():
    pair_selector = (
        (mt.AutoNameBinPairSelector() & mt.LensBinPairSelector())
        | (
            mt.NamedBinPairSelector(names=[("bin_1", "bin_2")])
            & mt.SourceBinPairSelector()
        )
        | mt.FirstNeighborsBinPairSelector()
    )
    yaml_str = yaml.dump(
        pair_selector.model_dump(), sort_keys=False, default_flow_style=None
    )
    pair_selector_from_yaml = mt.BinPairSelector.model_validate(
        yaml.safe_load(yaml_str)
    )
    assert pair_selector == pair_selector_from_yaml
    assert pair_selector.model_dump() == pair_selector_from_yaml.model_dump()


def test_pair_selector_serialization_negation():
    base_rule = mt.AutoNameBinPairSelector()
    neg_rule = ~base_rule
    yaml_str = yaml.dump(neg_rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert neg_rule == rule_from_yaml
    assert neg_rule.model_dump() == rule_from_yaml.model_dump()


def test_pair_selector_serialization_double_negation():
    base_rule = mt.AutoNameBinPairSelector()
    double_neg_rule = ~~base_rule
    yaml_str = yaml.dump(double_neg_rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert double_neg_rule == rule_from_yaml
    assert double_neg_rule.model_dump() == rule_from_yaml.model_dump()


def test_pair_selector_serialization_and_not():
    rule = mt.AutoNameBinPairSelector() & ~mt.LensBinPairSelector()
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert rule == rule_from_yaml
    assert rule.model_dump() == rule_from_yaml.model_dump()


def test_pair_selector_serialization_or_not():
    rule = mt.AutoNameBinPairSelector() | ~mt.LensBinPairSelector()
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert rule == rule_from_yaml
    assert isinstance(rule_from_yaml, mt.OrBinPairSelector)
    assert rule.model_dump() == rule_from_yaml.model_dump()


def test_pair_selector_serialization_type_source():
    rule = mt.TypeSourceBinPairSelector(type_source=mt.TypeSource("NewTypeSource"))
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert rule == rule_from_yaml
    assert rule.model_dump() == rule_from_yaml.model_dump()
    assert isinstance(rule_from_yaml, mt.TypeSourceBinPairSelector)
    assert rule.type_source == rule_from_yaml.type_source


def test_pair_selector_deserialization_lens():
    yaml_str = """
    kind: lens
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.LensBinPairSelector)


def test_pair_selector_deserialization_source():
    yaml_str = """
    kind: source
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.SourceBinPairSelector)


def test_pair_selector_deserialization_named():
    yaml_str = """
    kind: named
    names:
      - [bin_1, bin_2]
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.NamedBinPairSelector)
    assert rule.names == [("bin_1", "bin_2")]


def test_pair_selector_deserialization_auto_name():
    yaml_str = """
    kind: auto-name
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.AutoNameBinPairSelector)


def test_pair_selector_deserialization_auto_measurement():
    yaml_str = """
    kind: auto-measurement
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.AutoMeasurementBinPairSelector)


def test_pair_selector_deserialization_not():
    yaml_str = """
    kind: not
    pair_selector:
      kind: lens
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.NotBinPairSelector)
    assert isinstance(rule.pair_selector, mt.LensBinPairSelector)


def test_pair_selector_deserialization_and():
    yaml_str = """
    kind: and
    pair_selectors:
      - kind: auto-name
      - kind: lens
      - kind: source
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.AndBinPairSelector)
    assert len(rule.pair_selectors) == 3
    assert isinstance(rule.pair_selectors[0], mt.AutoNameBinPairSelector)
    assert isinstance(rule.pair_selectors[1], mt.LensBinPairSelector)
    assert isinstance(rule.pair_selectors[2], mt.SourceBinPairSelector)


def test_pair_selector_deserialization_or():
    yaml_str = """
    kind: or
    pair_selectors:
      - kind: auto-name
      - kind: lens
      - kind: source
      - kind: auto-measurement
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.OrBinPairSelector)
    assert len(rule.pair_selectors) == 4
    assert isinstance(rule.pair_selectors[0], mt.AutoNameBinPairSelector)
    assert isinstance(rule.pair_selectors[1], mt.LensBinPairSelector)
    assert isinstance(rule.pair_selectors[2], mt.SourceBinPairSelector)
    assert isinstance(rule.pair_selectors[3], mt.AutoMeasurementBinPairSelector)


def test_pair_selector_deserialization_nested_and_or():
    yaml_str = """
    kind: and
    pair_selectors:
      - kind: or
        pair_selectors:
          - kind: auto-name
          - kind: lens
          - kind: source
      - kind: auto-measurement
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.AndBinPairSelector)
    assert len(rule.pair_selectors) == 2
    assert isinstance(rule.pair_selectors[0], mt.OrBinPairSelector)
    assert len(rule.pair_selectors[0].pair_selectors) == 3
    assert isinstance(
        rule.pair_selectors[0].pair_selectors[0], mt.AutoNameBinPairSelector
    )
    assert isinstance(rule.pair_selectors[0].pair_selectors[1], mt.LensBinPairSelector)
    assert isinstance(
        rule.pair_selectors[0].pair_selectors[2], mt.SourceBinPairSelector
    )
    assert isinstance(rule.pair_selectors[1], mt.AutoMeasurementBinPairSelector)


def test_pair_selector_deserialization_nested_not():
    yaml_str = """
    kind: not
    pair_selector:
      kind: or
      pair_selectors:
        - kind: auto-name
        - kind: lens
        - kind: source
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.NotBinPairSelector)
    assert isinstance(rule.pair_selector, mt.OrBinPairSelector)
    assert len(rule.pair_selector.pair_selectors) == 3
    assert isinstance(rule.pair_selector.pair_selectors[0], mt.AutoNameBinPairSelector)
    assert isinstance(rule.pair_selector.pair_selectors[1], mt.LensBinPairSelector)
    assert isinstance(rule.pair_selector.pair_selectors[2], mt.SourceBinPairSelector)


def test_pair_selector_deserialization_nested_or_and():
    yaml_str = """
    kind: or
    pair_selectors:
      - kind: and
        pair_selectors:
          - kind: auto-name
          - kind: lens
          - kind: source
      - kind: auto-measurement
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.OrBinPairSelector)
    assert len(rule.pair_selectors) == 2
    assert isinstance(rule.pair_selectors[0], mt.AndBinPairSelector)
    assert len(rule.pair_selectors[0].pair_selectors) == 3
    assert isinstance(
        rule.pair_selectors[0].pair_selectors[0], mt.AutoNameBinPairSelector
    )
    assert isinstance(rule.pair_selectors[0].pair_selectors[1], mt.LensBinPairSelector)
    assert isinstance(
        rule.pair_selectors[0].pair_selectors[2], mt.SourceBinPairSelector
    )
    assert isinstance(rule.pair_selectors[1], mt.AutoMeasurementBinPairSelector)


def test_pair_selector_deserialization_invalid_kind():
    yaml_str = """
    kind: not_a_valid_kind
    """
    with pytest.raises(ValueError, match="Value error, Unknown kind not_a_valid_kind"):
        mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
