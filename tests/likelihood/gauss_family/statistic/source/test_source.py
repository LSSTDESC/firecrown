"""
Tests for the module firecrown.likelihood.statistic.source.
"""

# pylint: disable=too-many-locals

from typing import List, Callable
import itertools as it

import numpy as np
import pyccl
import pytest
import sacc
import numpy.typing as npt
from numpy.testing import assert_allclose
from scipy.interpolate import Akima1DInterpolator

import firecrown.likelihood.number_counts as nc
import firecrown.likelihood.number_counts._factories as nc_factories
import firecrown.likelihood.number_counts._systematics as nc_sys
import firecrown.likelihood._weak_lensing as wl
from firecrown.likelihood.number_counts._args import NumberCountsArgs
from firecrown.likelihood._weak_lensing import WeakLensingArgs
from firecrown.likelihood._source import (
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxySelectField,
    Tracer,
    dndz_shift_and_stretch_active,
    dndz_shift_and_stretch_passive,
)
from firecrown.metadata_functions import extract_all_tracers_inferred_galaxy_zdists
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap
from firecrown.updatable import get_default_params


@pytest.fixture(
    name="nc_sys_factory",
    params=[
        nc_factories.PhotoZShiftFactory(),
        nc_factories.PhotoZShiftandStretchFactory(),
        nc_factories.LinearBiasSystematicFactory(),
        nc_factories.PTNonLinearBiasSystematicFactory(),
        nc_factories.MagnificationBiasSystematicFactory(),
        nc_factories.ConstantMagnificationBiasSystematicFactory(),
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
def fixture_nc_sys_factory(request) -> nc_factories.NumberCountsSystematicFactory:
    """Fixture for the NumberCountsSystematicFactory class."""
    return request.param


@pytest.fixture(
    name="wl_sys_factory",
    params=[
        wl.LinearAlignmentSystematicFactory(),
        wl.MultiplicativeShearBiasFactory(),
        wl.TattAlignmentSystematicFactory(include_z_dependence=True),
        wl.TattAlignmentSystematicFactory(include_z_dependence=False),
        wl.PhotoZShiftFactory(),
        wl.PhotoZShiftandStretchFactory(),
    ],
    ids=[
        "LinearAlignmentSystematicFactory",
        "MultiplicativeShearBiasFactory",
        "TattAlignmentSystematicFactory_z",
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
    a = nc_sys.LinearBiasSystematic("xxx")
    assert isinstance(a, nc_sys.LinearBiasSystematic)
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
    a = nc_sys.LinearBiasSystematic("xxx")
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

    source_ready = wl.WeakLensing.create_ready(tomographic_bin=src0)

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
    source_ready = wl_factory.create(tomographic_bin=src0)

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
    source_ready = wl_factory.create(tomographic_bin=src0)

    assert source_ready is wl_factory.create(tomographic_bin=src0)


@pytest.mark.parametrize("include_z_dependence", [True, False])
def test_weak_lensing_source_factory_global_systematics(
    sacc_galaxy_cells_src0_src0, include_z_dependence
):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    src0 = next((obj for obj in all_tracers if obj.bin_name == "src0"), None)
    assert src0 is not None

    global_systematics: List[
        wl.LinearAlignmentSystematicFactory | wl.TattAlignmentSystematicFactory
    ] = [
        wl.LinearAlignmentSystematicFactory(),
        wl.TattAlignmentSystematicFactory(include_z_dependence=include_z_dependence),
    ]
    wl_factory = wl.WeakLensingFactory(
        per_bin_systematics=[], global_systematics=global_systematics
    )
    source_ready = wl_factory.create(tomographic_bin=src0)

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

    source_ready = nc.NumberCounts.create_ready(tomographic_bin=lens0)

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
    source_ready = nc_factory.create(tomographic_bin=lens0)

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
    source_ready = nc_factory.create(tomographic_bin=lens0)

    assert source_ready is nc_factory.create(tomographic_bin=lens0)


def test_number_counts_source_factory_global_systematics(sacc_galaxy_cells_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_cells_lens0_lens0

    all_tracers = extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    lens0 = next((obj for obj in all_tracers if obj.bin_name == "lens0"), None)
    assert lens0 is not None

    global_systematics = [nc_factories.PTNonLinearBiasSystematicFactory()]
    nc_factory = nc.NumberCountsFactory(
        per_bin_systematics=[],
        global_systematics=global_systematics,
    )
    source_ready = nc_factory.create(tomographic_bin=lens0)

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
    nc_sys_factory: nc_factories.NumberCountsSystematicFactory,
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
    factory = nc_factories.PhotoZShiftFactory()
    with pytest.raises(ValueError, match="PhotoZShift cannot be global"):
        _ = factory.create_global()


def test_nc_photozshiftandstretchfactory_no_globals():
    factory = nc_factories.PhotoZShiftandStretchFactory()
    with pytest.raises(ValueError, match="PhotoZShiftandStretch cannot be global"):
        _ = factory.create_global()


def test_nc_linearbiassystematicfactory_no_globals():
    factory = nc_factories.LinearBiasSystematicFactory()
    with pytest.raises(ValueError, match="LinearBiasSystematic cannot be global"):
        _ = factory.create_global()


def test_nc_magnificationbiassystematicfactory_no_globals():
    factory = nc_factories.MagnificationBiasSystematicFactory()
    with pytest.raises(
        ValueError, match="MagnificationBiasSystematic cannot be global"
    ):
        _ = factory.create_global()


def test_nc_constantmagnificationbiassystematicfactory_no_globals():
    factory = nc_factories.ConstantMagnificationBiasSystematicFactory()
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
    a = nc_sys.PhotoZShiftandStretch("xxx")
    assert isinstance(a, nc_sys.PhotoZShiftandStretch)
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


@pytest.mark.parametrize(
    "shift,stretch,transform",
    it.product(
        [0.0, 0.1, -0.1],
        [1.0, 0.9, 1.1],
        [dndz_shift_and_stretch_active, dndz_shift_and_stretch_passive],
    ),
)
def test_dndz_shift_active(
    shift: float,
    stretch: float,
    transform: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], float, float],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ],
):
    sigma = 0.05
    mu = 0.5
    shift = 0.1
    stretch = 1.0

    z_array = np.linspace(0.2, 1.2, 20_000, dtype=np.float64)
    dndz_array = (
        np.exp(-0.5 * (z_array - mu) ** 2 / sigma**2) / sigma / np.sqrt(2.0 * np.pi)
    )

    dndz_shifted_stretched = (
        np.exp(-0.5 * (z_array - mu + shift) ** 2 / (stretch * sigma) ** 2)
        / sigma
        / stretch
        / np.sqrt(2.0 * np.pi)
    )

    z_new, dndz_new = transform(z_array, dndz_array, shift, stretch)

    dndz_interp = Akima1DInterpolator(z_new, dndz_new, method="makima")
    dndz_interp_vals = np.nan_to_num(dndz_interp(z_array))

    assert_allclose(dndz_interp_vals, dndz_shifted_stretched, rtol=1.0e-6, atol=1.0e-6)


def test_dndz_shift_and_stretch_active_negative_sigma():
    with pytest.raises(ValueError, match="Stretch Parameter must be positive"):
        dndz_shift_and_stretch_active(
            z=np.array([0.0, 0.1]),
            dndz=np.array([1.0, 1.0]),
            delta_z=0.0,
            sigma_z=-1.0,
        )


def test_dndz_shift_and_stretch_passive_negative_sigma():
    with pytest.raises(ValueError, match="Stretch Parameter must be positive"):
        dndz_shift_and_stretch_passive(
            z=np.array([0.0, 0.1]),
            dndz=np.array([1.0, 1.0]),
            delta_z=0.0,
            sigma_z=-1.0,
        )


@pytest.mark.parametrize("shift,active", it.product([0.0, 0.1, -0.1], [True, False]))
def test_photoz_shift(
    tools_with_vanilla_cosmology: ModelingTools, shift: float, active: bool
):
    photoz_shift = nc_sys.PhotoZShift("John", active=active)
    photoz_shift.update(ParamsMap({"John_delta_z": shift}))
    sigma = 0.05
    mu = 0.5

    z_array = np.linspace(0.2, 1.2, 20_000, dtype=np.float64)
    dndz_array = (
        np.exp(-0.5 * (z_array - mu) ** 2 / sigma**2) / sigma / np.sqrt(2.0 * np.pi)
    )
    tracer_args = NumberCountsArgs(z=z_array, dndz=dndz_array)
    mod_tracer_args = photoz_shift.apply(tools_with_vanilla_cosmology, tracer_args)

    if active:
        mod_z, mod_dndz = dndz_shift_and_stretch_active(z_array, dndz_array, shift, 1.0)
    else:
        mod_z, mod_dndz = dndz_shift_and_stretch_passive(
            z_array, dndz_array, shift, 1.0
        )

    assert_allclose(mod_z, mod_tracer_args.z)
    assert_allclose(mod_dndz, mod_tracer_args.dndz)


@pytest.mark.parametrize(
    "shift,stretch,active", it.product([0.0, 0.1, -0.1], [1.0, 0.9, 1.1], [True, False])
)
def test_photoz_shift_stretch(
    tools_with_vanilla_cosmology: ModelingTools,
    shift: float,
    stretch: float,
    active: bool,
):
    photoz_shift_stretch = nc_sys.PhotoZShiftandStretch("John", active=active)
    photoz_shift_stretch.update(
        ParamsMap({"John_delta_z": shift, "John_sigma_z": stretch})
    )
    sigma = 0.05
    mu = 0.5

    z_array = np.linspace(0.2, 1.2, 20_000, dtype=np.float64)
    dndz_array = (
        np.exp(-0.5 * (z_array - mu) ** 2 / sigma**2) / sigma / np.sqrt(2.0 * np.pi)
    )
    tracer_args = NumberCountsArgs(z=z_array, dndz=dndz_array)
    mod_tracer_args = photoz_shift_stretch.apply(
        tools_with_vanilla_cosmology, tracer_args
    )

    if active:
        mod_z, mod_dndz = dndz_shift_and_stretch_active(
            z_array, dndz_array, shift, stretch
        )
    else:
        mod_z, mod_dndz = dndz_shift_and_stretch_passive(
            z_array, dndz_array, shift, stretch
        )

    assert_allclose(mod_z, mod_tracer_args.z)
    assert_allclose(mod_dndz, mod_tracer_args.dndz)


def _check_tatt_alignment_systematic_zdep(
    sys: wl.TattAlignmentSystematic, include_z_dependence: bool
):
    """Check that the TattAlignmentSystematic has the expected z-dependence
    parameters.
    """
    z_params = [
        "ia_zpiv_1",
        "ia_alphaz_1",
        "ia_zpiv_2",
        "ia_alphaz_2",
        "ia_zpiv_d",
        "ia_alphaz_d",
    ]
    default_params = get_default_params(sys)
    if include_z_dependence:
        for param in z_params:
            assert param in default_params
    else:
        for param in z_params:
            assert param not in default_params


@pytest.mark.parametrize("include_z_dependence", [True, False])
def test_tatt_alignment_systematics(include_z_dependence):
    sys = wl.TattAlignmentSystematic(include_z_dependence=include_z_dependence)
    assert isinstance(sys, wl.TattAlignmentSystematic)
    _check_tatt_alignment_systematic_zdep(
        sys, include_z_dependence=include_z_dependence
    )


@pytest.mark.parametrize("include_z_dependence", [True, False])
def test_tatt_alignment_systematic_factory(include_z_dependence):
    factory = wl.TattAlignmentSystematicFactory(
        include_z_dependence=include_z_dependence
    )
    sys = factory.create_global()
    assert isinstance(sys, wl.TattAlignmentSystematic)
    _check_tatt_alignment_systematic_zdep(
        sys, include_z_dependence=include_z_dependence
    )
