"""
Tests for the module firecrown.likelihood.statistic.source.
"""

# pylint: disable=too-many-locals

from typing import List

import numpy as np
import pyccl
import pytest
import sacc
from numpy.testing import assert_allclose

import firecrown.likelihood.number_counts as nc
import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood.number_counts import NumberCountsArgs
from firecrown.likelihood.weak_lensing import WeakLensingArgs
from firecrown.likelihood.source import (
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxySelectField,
    Tracer,
)
from firecrown.metadata_functions import extract_all_tracers_inferred_galaxy_zdists
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap


@pytest.fixture(
    name="nc_sys_factory",
    params=[
        nc.PhotoZShiftFactory(),
        nc.PhotoZShiftandStretchFactory(),
        nc.LinearBiasSystematicFactory(),
        nc.PTNonLinearBiasSystematicFactory(),
        nc.MagnificationBiasSystematicFactory(),
        nc.ConstantMagnificationBiasSystematicFactory(),
    ],
    ids=[
        "PhotoZShiftFactory",
        "PhotoZShiftandStretchFactory",
        "LinearBiasSystematicFactory",
        "PTNonLinearBiasSystematicFactory",
        "MagnificationBiasSystematicFactory",
        "ConstantMagnificationBiasSystematicFactory",
    ],
)
def fixture_nc_sys_factory(request) -> nc.NumberCountsSystematicFactory:
    """Fixture for the NumberCountsSystematicFactory class."""
    return request.param


@pytest.fixture(
    name="wl_sys_factory",
    params=[
        wl.LinearAlignmentSystematicFactory(),
        wl.MultiplicativeShearBiasFactory(),
        wl.TattAlignmentSystematicFactory(),
        wl.PhotoZShiftFactory(),
        wl.PhotoZShiftandStretchFactory(),
    ],
    ids=[
        "LinearAlignmentSystematicFactory",
        "MultiplicativeShearBiasFactory",
        "TattAlignmentSystematicFactory",
        "PhotoZShiftFactory",
        "PhotoZShiftandStretchFactory",
    ],
)
def fixture_wl_sys_factory(request):
    """Fixture for the WeakLensingSystematicFactory class."""
    return request.param


class TrivialSourceGalaxyArgs(SourceGalaxyArgs):
    """This is the most trivial possible subclass of SourceGalaxyArgs."""


class TrivialSourceGalaxy(SourceGalaxy[TrivialSourceGalaxyArgs]):
    """This is the most trivial possible subclass of SourceGalaxy."""

    def create_tracers(self, tools: ModelingTools):
        return Tracer(pyccl.Tracer()), TrivialSourceGalaxyArgs(
            z=np.array([]), dndz=np.array([])
        )

    def get_scale(self) -> float:
        return 1.0


def test_trivial_tracer_construction(empty_pyccl_tracer):
    trivial = Tracer(empty_pyccl_tracer)
    assert trivial.ccl_tracer is empty_pyccl_tracer
    # If we have not assigned a name through the initializer, the name of the resulting
    # Tracer object is the class name of the :class:`pyccl.Tracer` that was used to
    # create it.
    assert trivial.tracer_name == "Tracer"
    # If we have not supplied a field, and have not supplied a tracer_name, then the
    # field is "delta_matter".
    assert trivial.field == "delta_matter"
    assert trivial.pt_tracer is None
    assert not trivial.has_pt


def test_tracer_construction_with_name(empty_pyccl_tracer):
    named = Tracer(empty_pyccl_tracer, tracer_name="Fred")
    assert named.ccl_tracer is empty_pyccl_tracer
    assert named.tracer_name == "Fred"
    assert named.field == "Fred"
    assert named.pt_tracer is None
    assert not named.has_pt


def test_nc_linear_bias_systematic():
    a = nc.LinearBiasSystematic("xxx")
    assert isinstance(a, nc.LinearBiasSystematic)
    assert a.parameter_prefix == "xxx"
    assert a.alphag is None
    assert a.alphaz is None
    assert a.z_piv is None
    assert not a.is_updated()

    a.update(ParamsMap({"xxx_alphag": 1.0, "xxx_alphaz": 2.0, "xxx_z_piv": 1.5}))
    assert a.is_updated()
    assert a.alphag == 1.0
    assert a.alphaz == 2.0
    assert a.z_piv == 1.5

    a.reset()
    assert not a.is_updated()
    assert a.parameter_prefix == "xxx"
    assert a.alphag is None
    assert a.alphaz is None
    assert a.z_piv is None


def test_nc_nonlinear_bias_systematic_tracer_args_missing(
    tools_with_vanilla_cosmology: ModelingTools,
):
    a = nc.LinearBiasSystematic("xxx")
    # The values in the ParamsMap and the tracer_args are set to allow easy verification
    # that a tracer_args of None is handled correctly.
    a.update(ParamsMap({"xxx_alphag": 1.0, "xxx_alphaz": 1.0, "xxx_z_piv": 0.0}))
    tracer_args = NumberCountsArgs(z=np.array([0.0]), dndz=np.array([1.0]))
    new_tracer_args = a.apply(tools_with_vanilla_cosmology, tracer_args)
    assert new_tracer_args.bias is not None
    assert np.allclose(new_tracer_args.bias, np.array([1.0]))


def test_trivial_source_galaxy_construction():
    trivial = TrivialSourceGalaxy(sacc_tracer="no-sacc-tracer")

    with pytest.raises(
        RuntimeError,
        match="Must initialize tracer_args before calling _read on SourceGalaxy",
    ):
        trivial.read(sacc.Sacc())


def test_trivial_source_select_field():
    tools = ModelingTools()
    trivial = TrivialSourceGalaxy(sacc_tracer="no-sacc-tracer")
    select_field: SourceGalaxySelectField = SourceGalaxySelectField("new_field")
    trivial.tracer_args = TrivialSourceGalaxyArgs(
        z=np.array([1.0]), dndz=np.array([1.0]), field="old_field"
    )

    new_args = select_field.apply(tools, trivial.tracer_args)
    assert new_args.field == "new_field"
    assert trivial.tracer_args.field == "old_field"


def test_weak_lensing_source_init(
    sacc_galaxy_cells_src0_src0, tools_with_vanilla_cosmology: ModelingTools
):
    sacc_data, z, dndz = sacc_galaxy_cells_src0_src0

    source = wl.WeakLensing(sacc_tracer="src0")
    source.read(sacc_data)

    assert_allclose(source.tracer_args.z, z)
    assert_allclose(source.tracer_args.dndz, dndz)
    source.update(ParamsMap())
    ts = source.get_tracers(tools_with_vanilla_cosmology)
    assert len(ts) == 1
    dp = source.get_derived_parameters()
    assert dp is not None
    assert len(dp) == 0


def test_weak_lensing_source_create_ready(sacc_galaxy_cells_src0_src0):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    src0 = next((obj for obj in all_tracers if obj.bin_name == "src0"), None)
    assert src0 is not None

    source_ready = wl.WeakLensing.create_ready(inferred_zdist=src0)

    source_read = wl.WeakLensing(sacc_tracer="src0")
    source_read.read(sacc_data)

    assert_allclose(source_ready.tracer_args.z, source_read.tracer_args.z)
    assert_allclose(source_ready.tracer_args.dndz, source_read.tracer_args.dndz)


def test_weak_lensing_source_factory(sacc_galaxy_cells_src0_src0):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    src0 = next((obj for obj in all_tracers if obj.bin_name == "src0"), None)
    assert src0 is not None

    wl_factory = wl.WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    source_ready = wl_factory.create(inferred_zdist=src0)

    source_read = wl.WeakLensing(sacc_tracer="src0")
    source_read.read(sacc_data)

    assert_allclose(source_ready.tracer_args.z, source_read.tracer_args.z)
    assert_allclose(source_ready.tracer_args.dndz, source_read.tracer_args.dndz)


def test_weak_lensing_source_factory_cache(sacc_galaxy_cells_src0_src0):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    src0 = next((obj for obj in all_tracers if obj.bin_name == "src0"), None)
    assert src0 is not None

    wl_factory = wl.WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    source_ready = wl_factory.create(inferred_zdist=src0)

    assert source_ready is wl_factory.create(inferred_zdist=src0)


def test_weak_lensing_source_factory_global_systematics(sacc_galaxy_cells_src0_src0):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    src0 = next((obj for obj in all_tracers if obj.bin_name == "src0"), None)
    assert src0 is not None

    global_systematics: List[
        wl.LinearAlignmentSystematicFactory | wl.TattAlignmentSystematicFactory
    ] = [
        wl.LinearAlignmentSystematicFactory(),
        wl.TattAlignmentSystematicFactory(),
    ]
    wl_factory = wl.WeakLensingFactory(
        per_bin_systematics=[], global_systematics=global_systematics
    )
    source_ready = wl_factory.create(inferred_zdist=src0)

    # pylint: disable=protected-access
    source_read = wl.WeakLensing(
        sacc_tracer="src0", systematics=wl_factory._global_systematics_instances
    )
    # pylint: enable=protected-access
    source_read.read(sacc_data)

    assert_allclose(source_ready.tracer_args.z, source_read.tracer_args.z)
    assert_allclose(source_ready.tracer_args.dndz, source_read.tracer_args.dndz)
    assert source_ready.systematics == source_read.systematics


def test_weak_lensing_source_init_wrong_name(sacc_galaxy_cells_src0_src0):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    source = wl.WeakLensing(sacc_tracer="src10")
    with pytest.raises(KeyError, match="src10"):
        source.read(sacc_data)


def test_number_counts_source_init(
    sacc_galaxy_cells_lens0_lens0, tools_with_vanilla_cosmology: ModelingTools
):

    sacc_data, z, dndz = sacc_galaxy_cells_lens0_lens0

    source = nc.NumberCounts(sacc_tracer="lens0")
    source.read(sacc_data)

    assert_allclose(source.tracer_args.z, z)
    assert_allclose(source.tracer_args.dndz, dndz)
    source.update(ParamsMap(lens0_bias=1.1))
    dp = source.get_derived_parameters()
    assert dp is not None
    assert len(dp) == 0
    ts = source.get_tracers(tools_with_vanilla_cosmology)
    assert len(ts) == 1


def test_number_counts_source_create_ready(sacc_galaxy_cells_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_cells_lens0_lens0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    lens0 = next((obj for obj in all_tracers if obj.bin_name == "lens0"), None)
    assert lens0 is not None

    source_ready = nc.NumberCounts.create_ready(inferred_zdist=lens0)

    source_read = nc.NumberCounts(sacc_tracer="lens0")
    source_read.read(sacc_data)

    assert_allclose(source_ready.tracer_args.z, source_read.tracer_args.z)
    assert_allclose(source_ready.tracer_args.dndz, source_read.tracer_args.dndz)


def test_number_counts_source_factory(sacc_galaxy_cells_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_cells_lens0_lens0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    lens0 = next((obj for obj in all_tracers if obj.bin_name == "lens0"), None)
    assert lens0 is not None

    nc_factory = nc.NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    source_ready = nc_factory.create(inferred_zdist=lens0)

    source_read = nc.NumberCounts(sacc_tracer="lens0")
    source_read.read(sacc_data)

    assert_allclose(source_ready.tracer_args.z, source_read.tracer_args.z)
    assert_allclose(source_ready.tracer_args.dndz, source_read.tracer_args.dndz)


def test_number_counts_source_factory_cache(sacc_galaxy_cells_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_cells_lens0_lens0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    lens0 = next((obj for obj in all_tracers if obj.bin_name == "lens0"), None)
    assert lens0 is not None

    nc_factory = nc.NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    source_ready = nc_factory.create(inferred_zdist=lens0)

    assert source_ready is nc_factory.create(inferred_zdist=lens0)


def test_number_counts_source_factory_global_systematics(sacc_galaxy_cells_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_cells_lens0_lens0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    lens0 = next((obj for obj in all_tracers if obj.bin_name == "lens0"), None)
    assert lens0 is not None

    global_systematics = [nc.PTNonLinearBiasSystematicFactory()]
    nc_factory = nc.NumberCountsFactory(
        per_bin_systematics=[],
        global_systematics=global_systematics,
    )
    source_ready = nc_factory.create(inferred_zdist=lens0)

    # pylint: disable=protected-access
    source_read = nc.NumberCounts(
        sacc_tracer="lens0", systematics=nc_factory._global_systematics_instances
    )
    # pylint: enable=protected-access
    source_read.read(sacc_data)

    assert_allclose(source_ready.tracer_args.z, source_read.tracer_args.z)
    assert_allclose(source_ready.tracer_args.dndz, source_read.tracer_args.dndz)
    assert source_ready.systematics == source_read.systematics


def test_number_counts_source_init_wrong_name(sacc_galaxy_cells_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_cells_lens0_lens0

    source = nc.NumberCounts(sacc_tracer="lens10")
    with pytest.raises(KeyError, match="lens10"):
        source.read(sacc_data)


def test_number_counts_systematic_factory(
    nc_sys_factory: nc.NumberCountsSystematicFactory,
):
    sys_pz_shift = nc_sys_factory.create("bin_1")
    assert sys_pz_shift.parameter_prefix == "bin_1"


def test_wl_photozshiftfactory_no_globals():
    factory = wl.PhotoZShiftFactory()
    with pytest.raises(ValueError, match="PhotoZShift cannot be global"):
        _ = factory.create_global()


def test_wl_photozshiftandstretchfactory_no_globals():
    factory = wl.PhotoZShiftandStretchFactory()
    with pytest.raises(ValueError, match="PhotoZShiftandStretch cannot be global"):
        _ = factory.create_global()


def test_wl_multiplicativeshearbiasfactory_no_globals():
    factory = wl.MultiplicativeShearBiasFactory()
    with pytest.raises(ValueError, match="MultiplicativeShearBias cannot be global"):
        _ = factory.create_global()


def test_nc_photozshiftfactory_no_globals():
    factory = nc.PhotoZShiftFactory()
    with pytest.raises(ValueError, match="PhotoZShift cannot be global"):
        _ = factory.create_global()


def test_nc_photozshiftandstretchfactory_no_globals():
    factory = nc.PhotoZShiftandStretchFactory()
    with pytest.raises(ValueError, match="PhotoZShiftandStretch cannot be global"):
        _ = factory.create_global()


def test_nc_linearbiassystematicfactory_no_globals():
    factory = nc.LinearBiasSystematicFactory()
    with pytest.raises(ValueError, match="LinearBiasSystematic cannot be global"):
        _ = factory.create_global()


def test_nc_magnificationbiassystematicfactory_no_globals():
    factory = nc.MagnificationBiasSystematicFactory()
    with pytest.raises(
        ValueError, match="MagnificationBiasSystematic cannot be global"
    ):
        _ = factory.create_global()


def test_nc_constantmagnificationbiassystematicfactory_no_globals():
    factory = nc.ConstantMagnificationBiasSystematicFactory()
    with pytest.raises(
        ValueError, match="ConstantMagnificationBiasSystematic cannot be global"
    ):
        _ = factory.create_global()


def test_weak_lensing_systematic_factory(
    wl_sys_factory: wl.WeakLensingSystematicFactory,
):
    sys_pz_shift = wl_sys_factory.create("bin_1")
    assert sys_pz_shift.parameter_prefix == "bin_1"


def test_wl_photozshiftandstretch_systematic(
    tools_with_vanilla_cosmology: ModelingTools,
):
    a = wl.PhotoZShiftandStretch("xxx")
    assert isinstance(a, wl.PhotoZShiftandStretch)
    assert a.parameter_prefix == "xxx"
    assert a.delta_z is None
    assert a.sigma_z is None
    assert not a.is_updated()

    a.update(ParamsMap({"xxx_delta_z": 0.0, "xxx_sigma_z": 1.0}))
    assert a.is_updated()
    assert a.delta_z == 0.0
    assert a.sigma_z == 1.0

    a.reset()
    assert not a.is_updated()
    assert a.parameter_prefix == "xxx"
    assert a.delta_z is None
    assert a.sigma_z is None

    a.update(ParamsMap({"xxx_delta_z": 0.0, "xxx_sigma_z": -1.0}))
    assert a.is_updated()
    tracer_args = WeakLensingArgs(z=np.array([0.0, 0.1]), dndz=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="Stretch Parameter must be positive"):
        _ = a.apply(tools_with_vanilla_cosmology, tracer_args)

    a.reset()
    a.update(ParamsMap({"xxx_delta_z": 0.0, "xxx_sigma_z": 1.0}))
    new_tracer_args = a.apply(tools_with_vanilla_cosmology, tracer_args)
    assert new_tracer_args.dndz is not None
    assert_allclose(new_tracer_args.z, tracer_args.z)
    assert_allclose(new_tracer_args.dndz, tracer_args.dndz)


def test_nc_photozshiftandstretch_systematic(
    tools_with_vanilla_cosmology: ModelingTools,
):
    a = nc.PhotoZShiftandStretch("xxx")
    assert isinstance(a, nc.PhotoZShiftandStretch)
    assert a.parameter_prefix == "xxx"
    assert a.delta_z is None
    assert a.sigma_z is None
    assert not a.is_updated()

    a.update(ParamsMap({"xxx_delta_z": 0.0, "xxx_sigma_z": 1.0}))
    assert a.is_updated()
    assert a.delta_z == 0.0
    assert a.sigma_z == 1.0

    a.reset()
    assert not a.is_updated()
    assert a.parameter_prefix == "xxx"
    assert a.delta_z is None
    assert a.sigma_z is None

    a.update(ParamsMap({"xxx_delta_z": 0.0, "xxx_sigma_z": -1.0}))
    assert a.is_updated()
    tracer_args = NumberCountsArgs(z=np.array([0.0, 0.1]), dndz=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="Stretch Parameter must be positive"):
        _ = a.apply(tools_with_vanilla_cosmology, tracer_args)

    a.reset()
    a.update(ParamsMap({"xxx_delta_z": 0.0, "xxx_sigma_z": 1.0}))
    new_tracer_args = a.apply(tools_with_vanilla_cosmology, tracer_args)
    assert new_tracer_args.dndz is not None
    assert_allclose(new_tracer_args.z, tracer_args.z)
    assert_allclose(new_tracer_args.dndz, tracer_args.dndz)
