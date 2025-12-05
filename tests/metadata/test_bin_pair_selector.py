"""Tests for the BinPairSelector classes in firecrown.metadata_types.

This module contains comprehensive tests for all bin pair selector implementations,
including atomic selectors, composite selectors, logical combinators, and serialization.
"""

from typing import Any
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
    first_neighbor_pair_selector = mt.AutoNameDiffBinPairSelector(
        neighbors_diff=[0, 1, -1]
    )

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
    first_neighbor_pair_selector = mt.AutoNameDiffBinPairSelector(
        neighbors_diff=[1, -1]
    )
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
    rule = mt.AutoNameDiffBinPairSelector(neighbors_diff=[0, 1, -1])
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
        | mt.AutoNameDiffBinPairSelector(neighbors_diff=[0, 1, -1])
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


# ============================================================================
# CompositeSelector and BadSelector Tests
# ============================================================================


def test_composite_selector_bad_selector_not_initialized():
    """Test that CompositeSelector raises NotImplementedError when _impl is not set."""
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

    # Create a composite selector that doesn't properly initialize _impl
    class BadCompositeSelector(mt.CompositeSelector):
        """A composite selector that doesn't initialize _impl."""

        kind: str = "bad-composite"

        def model_post_init(self, _: Any, /) -> None:
            # Intentionally not setting _impl - leaves it as BadSelector
            pass

    # Register temporarily for this test
    bad_composite = BadCompositeSelector()

    # Trying to use keep() should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="BadSelector should not be used"):
        bad_composite.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))


def test_auto_bin_pair_selector(all_harmonic_bins):
    """Test AutoBinPairSelector composite selector."""
    auto_pair_selector = mt.AutoBinPairSelector()

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, auto_pair_selector
    )
    # AutoBinPairSelector should create all auto-combinations
    # There are two measurements per bin in all_harmonic_bins
    assert len(two_point_xy_combinations) == 2 * len(all_harmonic_bins)
    for two_point_xy in two_point_xy_combinations:
        assert two_point_xy.x == two_point_xy.y
        assert two_point_xy.x_measurement == two_point_xy.y_measurement


def test_auto_bin_pair_selector_keep():
    """Test AutoBinPairSelector keep method."""
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
    rule = mt.AutoBinPairSelector()
    # Same bin and same measurement: should keep
    assert rule.keep((z1, z1), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    # Different bins: should not keep
    assert not rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    # Same bin but different measurements: should not keep
    assert not rule.keep((z1, z1), (mt.Galaxies.COUNTS, mt.Galaxies.SHEAR_E))


def test_source_lens_pair_selector():
    """Test SourceLensBinPairSelector with order-dependent matching."""
    source_lens_pair_selector = mt.SourceLensBinPairSelector()

    # Create a mixed measurement bin for testing
    z_source = mt.InferredGalaxyZDist(
        bin_name="src0",
        z=np.linspace(0, 1.0, 50) + 0.05,
        dndz=np.exp(-0.5 * (np.linspace(0, 1.0, 50) + 0.05 - 0.5) ** 2 / 0.05 / 0.05),
        measurements={mt.Galaxies.SHEAR_E},
    )
    z_lens = mt.InferredGalaxyZDist(
        bin_name="lens0",
        z=np.linspace(0, 1.0, 50) + 0.05,
        dndz=np.exp(-0.5 * (np.linspace(0, 1.0, 50) + 0.05 - 0.5) ** 2 / 0.05 / 0.05),
        measurements={mt.Galaxies.COUNTS},
    )

    mixed_bins = [z_source, z_lens]

    two_point_xy_combinations = make_binned_two_point_filtered(
        mixed_bins, source_lens_pair_selector
    )
    # SourceLensBinPairSelector should only keep source-left, lens-right pairs
    # Not lens-left, source-right (order matters)
    assert len(two_point_xy_combinations) == 1
    two_point_xy = two_point_xy_combinations[0]
    assert two_point_xy.x.bin_name == "src0"
    assert two_point_xy.y.bin_name == "lens0"
    assert two_point_xy.x_measurement == mt.Galaxies.SHEAR_E
    assert two_point_xy.y_measurement == mt.Galaxies.COUNTS


def test_source_lens_pair_selector_keep():
    """Test SourceLensBinPairSelector keep method with fixed order."""
    z_source = mt.InferredGalaxyZDist(
        bin_name="src1",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    z_lens = mt.InferredGalaxyZDist(
        bin_name="lens1",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )

    rule = mt.SourceLensBinPairSelector()
    # Source-left, lens-right should match
    assert rule.keep((z_source, z_lens), (mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS))
    # Lens-left, source-right should NOT match (fixed order convention)
    assert not rule.keep((z_lens, z_source), (mt.Galaxies.COUNTS, mt.Galaxies.SHEAR_E))
    # Both source should NOT match
    assert not rule.keep(
        (z_source, z_source), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E)
    )
    # Both lens should NOT match
    assert not rule.keep((z_lens, z_lens), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))


def test_auto_bin_pair_selector_serialization():
    """Test serialization of AutoBinPairSelector."""
    rule = mt.AutoBinPairSelector()
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule_from_yaml, mt.AutoBinPairSelector)
    assert rule == rule_from_yaml


def test_source_lens_pair_selector_serialization():
    """Test serialization of SourceLensBinPairSelector."""
    rule = mt.SourceLensBinPairSelector()
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule_from_yaml, mt.SourceLensBinPairSelector)
    assert rule == rule_from_yaml


def test_auto_bin_pair_selector_deserialization():
    """Test deserialization of AutoBinPairSelector."""
    yaml_str = """
    kind: auto-bin
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.AutoBinPairSelector)


def test_source_lens_pair_selector_deserialization():
    """Test deserialization of SourceLensBinPairSelector."""
    yaml_str = """
    kind: source-lens
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.SourceLensBinPairSelector)


# ============================================================================
# Cross Selector Tests
# ============================================================================


def test_cross_name_bin_pair_selector_keep():
    """Test CrossNameBinPairSelector keep method."""
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
    rule = mt.CrossNameBinPairSelector()
    # Different bin names: should keep
    assert rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    # Same bin name: should NOT keep
    assert not rule.keep((z1, z1), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))


def test_cross_name_bin_pair_selector(all_harmonic_bins):
    """Test CrossNameBinPairSelector filters out auto-name correlations."""
    cross_name_selector = mt.CrossNameBinPairSelector()

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, cross_name_selector
    )
    # Should exclude same-name pairs (bin_1, bin_1) and (bin_2, bin_2)
    # Should include cross pairs: (bin_1, bin_2) with all measurement combinations
    # With 2 bins and 2 measurements each: 2*2*2 - 2*2 = 8 - 4 = 4 cross-name pairs
    # Actually: (bin_1, bin_2) with 2x2=4 measurement combos = 4 pairs
    for two_point_xy in two_point_xy_combinations:
        assert two_point_xy.x.bin_name != two_point_xy.y.bin_name


def test_cross_measurement_bin_pair_selector_keep():
    """Test CrossMeasurementBinPairSelector keep method."""
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
        measurements={mt.Galaxies.SHEAR_E},
    )
    rule = mt.CrossMeasurementBinPairSelector()
    # Different measurements: should keep
    assert rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.SHEAR_E))
    # Same measurement: should NOT keep
    assert not rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))


def test_cross_measurement_bin_pair_selector(all_harmonic_bins):
    """Test CrossMeasurementBinPairSelector.

    Test filters out auto-measurement correlations."""
    cross_measurement_selector = mt.CrossMeasurementBinPairSelector()

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, cross_measurement_selector
    )
    # Should exclude same-measurement pairs
    # Should include only cross-measurement pairs
    for two_point_xy in two_point_xy_combinations:
        assert two_point_xy.x_measurement != two_point_xy.y_measurement


def test_cross_bin_pair_selector_keep():
    """Test CrossBinPairSelector keep method."""
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
    rule = mt.CrossBinPairSelector()
    # Different bins: should keep
    assert rule.keep((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    # Same bin and same measurement: should NOT keep
    assert not rule.keep((z1, z1), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    # Same bin but different measurement: should keep (because AutoBinPairSelector
    # requires both to match)
    assert rule.keep((z1, z1), (mt.Galaxies.COUNTS, mt.Galaxies.SHEAR_E))


def test_cross_bin_pair_selector(all_harmonic_bins):
    """Test CrossBinPairSelector filters out auto-bin correlations."""
    cross_bin_selector = mt.CrossBinPairSelector()

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins, cross_bin_selector
    )
    # Should exclude pairs where BOTH bin name and measurement match
    # All other combinations should be included
    for two_point_xy in two_point_xy_combinations:
        # Not both bin name and measurement can be the same
        is_auto_bin = (
            two_point_xy.x.bin_name == two_point_xy.y.bin_name
            and two_point_xy.x_measurement == two_point_xy.y_measurement
        )
        assert not is_auto_bin


def test_cross_name_bin_pair_selector_serialization():
    """Test serialization of CrossNameBinPairSelector."""
    rule = mt.CrossNameBinPairSelector()
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule_from_yaml, mt.CrossNameBinPairSelector)
    assert rule == rule_from_yaml


def test_cross_measurement_bin_pair_selector_serialization():
    """Test serialization of CrossMeasurementBinPairSelector."""
    rule = mt.CrossMeasurementBinPairSelector()
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule_from_yaml, mt.CrossMeasurementBinPairSelector)
    assert rule == rule_from_yaml


def test_cross_bin_pair_selector_serialization():
    """Test serialization of CrossBinPairSelector."""
    rule = mt.CrossBinPairSelector()
    yaml_str = yaml.dump(rule.model_dump(), sort_keys=False)
    rule_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule_from_yaml, mt.CrossBinPairSelector)
    assert rule == rule_from_yaml


def test_cross_name_bin_pair_selector_deserialization():
    """Test deserialization of CrossNameBinPairSelector."""
    yaml_str = """
    kind: cross-name
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.CrossNameBinPairSelector)


def test_cross_measurement_bin_pair_selector_deserialization():
    """Test deserialization of CrossMeasurementBinPairSelector."""
    yaml_str = """
    kind: cross-measurement
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.CrossMeasurementBinPairSelector)


def test_cross_bin_pair_selector_deserialization():
    """Test deserialization of CrossBinPairSelector."""
    yaml_str = """
    kind: cross-bin
    """
    rule = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(rule, mt.CrossBinPairSelector)


def test_cross_selectors_are_inverses():
    """Test that Cross selectors are exact inverses of Auto selectors."""
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
        measurements={mt.Galaxies.SHEAR_E},
    )

    # Test all combinations
    test_cases = [
        ((z1, z1), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS)),
        ((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS)),
        ((z1, z2), (mt.Galaxies.COUNTS, mt.Galaxies.SHEAR_E)),
        ((z2, z1), (mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS)),
    ]

    # CrossName vs AutoName
    auto_name = mt.AutoNameBinPairSelector()
    cross_name = mt.CrossNameBinPairSelector()
    for zdist, measurements in test_cases:
        assert auto_name.keep(zdist, measurements) != cross_name.keep(
            zdist, measurements
        )

    # CrossMeasurement vs AutoMeasurement
    auto_measurement = mt.AutoMeasurementBinPairSelector()
    cross_measurement = mt.CrossMeasurementBinPairSelector()
    for zdist, measurements in test_cases:
        assert auto_measurement.keep(zdist, measurements) != cross_measurement.keep(
            zdist, measurements
        )

    # CrossBin vs AutoBin
    auto_bin = mt.AutoBinPairSelector()
    cross_bin = mt.CrossBinPairSelector()
    for zdist, measurements in test_cases:
        assert auto_bin.keep(zdist, measurements) != cross_bin.keep(zdist, measurements)


# ============================================================================
# 3x2pt Selector Tests
# ============================================================================


def test_three_two_bin_pair_selector_keep():
    """Test ThreeTwoBinPairSelector keep method with various combinations."""
    selector = mt.ThreeTwoBinPairSelector()

    # Create test bins
    src1 = mt.InferredGalaxyZDist(
        bin_name="src_0",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    src2 = mt.InferredGalaxyZDist(
        bin_name="src_1",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    lens1 = mt.InferredGalaxyZDist(
        bin_name="lens_0",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    lens2 = mt.InferredGalaxyZDist(
        bin_name="lens_2",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )

    # Test 1: Source-source (cosmic shear) - should be kept
    assert selector.keep((src1, src1), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert selector.keep((src1, src2), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))

    # Test 2: Lens-lens (galaxy clustering) - should be kept
    assert selector.keep((lens1, lens1), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    assert selector.keep((lens1, lens2), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))

    # Test 3: Source-lens with different names (galaxy-galaxy lensing) - should be kept
    assert selector.keep((src2, lens1), (mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS))

    # Test 4: Source-lens with SAME name - should NOT be kept (excluded by
    # CrossNameDiff)
    src_lens_same = mt.InferredGalaxyZDist(
        bin_name="bin_0",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
    )
    assert not selector.keep(
        (src_lens_same, src_lens_same), (mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS)
    )


def test_three_two_bin_pair_selector_integration(all_harmonic_bins):
    """Test ThreeTwoBinPairSelector with realistic bin structure."""
    selector = mt.ThreeTwoBinPairSelector()

    # Ensure we have distinct prefixes for cross source-lens pairs
    extra_bins = [
        mt.InferredGalaxyZDist(
            bin_name="srcX1",
            z=np.array([0.3]),
            dndz=np.array([1.0]),
            measurements={mt.Galaxies.SHEAR_E},
        ),
        mt.InferredGalaxyZDist(
            bin_name="lensX0",
            z=np.array([0.1]),
            dndz=np.array([1.0]),
            measurements={mt.Galaxies.COUNTS},
        ),
    ]

    two_point_xy_combinations = make_binned_two_point_filtered(
        all_harmonic_bins + extra_bins, selector
    )

    source_source_count = 0
    lens_lens_count = 0
    source_lens_cross_count = 0
    source_lens_auto_count = 0

    for two_point_xy in two_point_xy_combinations:
        x_is_source = two_point_xy.x_measurement in mt.GALAXY_SOURCE_TYPES
        y_is_source = two_point_xy.y_measurement in mt.GALAXY_SOURCE_TYPES
        x_is_lens = two_point_xy.x_measurement in mt.GALAXY_LENS_TYPES
        y_is_lens = two_point_xy.y_measurement in mt.GALAXY_LENS_TYPES
        same_bin_name = two_point_xy.x.bin_name == two_point_xy.y.bin_name

        if x_is_source and y_is_source:
            source_source_count += 1
        elif x_is_lens and y_is_lens:
            lens_lens_count += 1
        elif x_is_source and y_is_lens:
            if same_bin_name:
                source_lens_auto_count += 1
            else:
                source_lens_cross_count += 1

    # Verify we have all three components
    assert source_source_count > 0, "Should have cosmic shear pairs"
    assert lens_lens_count > 0, "Should have galaxy clustering pairs"
    assert source_lens_cross_count > 0, "Should have galaxy-galaxy lensing pairs"

    # Verify no auto-correlations for source-lens
    assert source_lens_auto_count == 0, "Should NOT have same-bin source-lens pairs"


def test_three_two_bin_pair_selector_components():
    """Test that ThreeTwoBinPairSelector includes the right components."""
    selector = mt.ThreeTwoBinPairSelector()

    # Explicitly build source-only and lens-only bins with distinct prefixes
    sources = [
        mt.InferredGalaxyZDist(
            bin_name=f"src{i}",
            z=np.array([0.1 * (i + 1)]),
            dndz=np.array([1.0]),
            measurements={mt.Galaxies.SHEAR_E},
        )
        for i in range(2)
    ]
    lenses = [
        mt.InferredGalaxyZDist(
            bin_name=f"lens{i}",
            z=np.array([0.2 * (i + 1)]),
            dndz=np.array([1.0]),
            measurements={mt.Galaxies.COUNTS},
        )
        for i in range(2)
    ]

    kept_pairs = []
    for i, bin1 in enumerate(sources + lenses):
        for j, bin2 in enumerate(sources + lenses):
            for meas1 in [mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS]:
                for meas2 in [mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS]:
                    if selector.keep((bin1, bin2), (meas1, meas2)):
                        kept_pairs.append((i, j, meas1, meas2))

    # Verify structure: should have source-source, lens-lens, and cross-name source-lens
    has_shear_shear = any(
        m1 == mt.Galaxies.SHEAR_E and m2 == mt.Galaxies.SHEAR_E
        for _, _, m1, m2 in kept_pairs
    )
    has_counts_counts = any(
        m1 == mt.Galaxies.COUNTS and m2 == mt.Galaxies.COUNTS
        for _, _, m1, m2 in kept_pairs
    )
    has_shear_counts_cross = any(
        m1 == mt.Galaxies.SHEAR_E and m2 == mt.Galaxies.COUNTS and i != j
        for i, j, m1, m2 in kept_pairs
    )
    has_shear_counts_auto = any(
        m1 == mt.Galaxies.SHEAR_E and m2 == mt.Galaxies.COUNTS and i == j
        for i, j, m1, m2 in kept_pairs
    )

    assert has_shear_shear, "Should include cosmic shear (source-source)"
    assert has_counts_counts, "Should include galaxy clustering (lens-lens)"
    assert has_shear_counts_cross, "Should include cross-bin galaxy-galaxy lensing"
    assert not has_shear_counts_auto, "Should NOT include same-bin source-lens"


def test_three_two_bin_pair_selector_serialization():
    """Test serialization of ThreeTwoBinPairSelector."""
    selector = mt.ThreeTwoBinPairSelector()
    yaml_str = yaml.dump(selector.model_dump(), sort_keys=False)
    selector_from_yaml = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(selector_from_yaml, mt.ThreeTwoBinPairSelector)
    assert selector == selector_from_yaml


def test_three_two_bin_pair_selector_deserialization():
    """Test deserialization of ThreeTwoBinPairSelector."""
    yaml_str = """
    kind: 3x2pt
    """
    selector = mt.BinPairSelector.model_validate(yaml.safe_load(yaml_str))
    assert isinstance(selector, mt.ThreeTwoBinPairSelector)


def test_three_two_bin_pair_selector_distance_parameters():
    """Test distance parameters for ThreeTwoBinPairSelector."""
    selector = mt.ThreeTwoBinPairSelector(
        source_dist=1, lens_dist=1, source_lens_dist=2
    )

    # Source bins with numeric suffixes
    src0 = mt.InferredGalaxyZDist(
        bin_name="src0",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    src1 = mt.InferredGalaxyZDist(
        bin_name="src1",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    src2 = mt.InferredGalaxyZDist(
        bin_name="src2",
        z=np.array([0.3]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    src3 = mt.InferredGalaxyZDist(
        bin_name="src3",
        z=np.array([0.4]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )

    # Lens bins
    lens0 = mt.InferredGalaxyZDist(
        bin_name="lens0",
        z=np.array([0.1]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    lens1 = mt.InferredGalaxyZDist(
        bin_name="lens1",
        z=np.array([0.2]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )
    lens3 = mt.InferredGalaxyZDist(
        bin_name="lens3",
        z=np.array([0.4]),
        dndz=np.array([1.0]),
        measurements={mt.Galaxies.COUNTS},
    )

    # Source-source: allowed diffs are -1 and 0 (range(-1,1))
    assert selector.keep((src0, src1), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert selector.keep((src0, src0), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert not selector.keep((src1, src0), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))
    assert not selector.keep((src0, src3), (mt.Galaxies.SHEAR_E, mt.Galaxies.SHEAR_E))

    # Lens-lens: allowed diffs are -1 and 0
    assert selector.keep((lens0, lens1), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    assert selector.keep((lens0, lens0), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    assert not selector.keep((lens1, lens0), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))
    assert not selector.keep((lens0, lens3), (mt.Galaxies.COUNTS, mt.Galaxies.COUNTS))

    # Source-lens: allowed diffs are 1..2 (positive only, different prefixes)
    assert selector.keep((src3, lens1), (mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS))
    assert selector.keep((src2, lens0), (mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS))
    assert not selector.keep((src0, lens1), (mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS))
