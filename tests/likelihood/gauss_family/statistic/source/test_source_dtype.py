"""Float64 contracts at public tomography and source ingress."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pytest
import sacc
import pyccl

import firecrown.likelihood.number_counts as nc
import firecrown.likelihood.weak_lensing as wl
from firecrown.metadata_functions import (
    extract_all_tracers_projected_fields,
    extract_all_tracers_tomographic_bins,
)
from firecrown.metadata_types import Galaxies, TomographicBin
from firecrown.likelihood_base import (
    SourceGalaxyArgs,
    dndz_shift_and_stretch_active,
    dndz_shift_and_stretch_passive,
)
from firecrown.likelihood.number_counts._args import NumberCountsArgs
from firecrown.likelihood.weak_lensing import WeakLensingArgs
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap

_NBit = TypeVar("_NBit", bound=npt.NBitBase)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_tomography_real_ingress(
    dtype: type[np.integer[_NBit] | np.floating[_NBit]],
) -> None:
    """Store real coordinates and unnormalized distributions as float64.

    :param dtype: Supported caller storage representation.
    """
    z: npt.NDArray[np.integer[_NBit] | np.floating[_NBit]] = np.array(
        [0, 1, 2, 3], dtype=dtype
    )
    dndz: npt.NDArray[np.integer[_NBit] | np.floating[_NBit]] = np.array(
        [0, 2, 1, 0], dtype=dtype
    )
    bin_ = TomographicBin(
        bin_name="src", z=z, dndz=dndz, measurements={Galaxies.SHEAR_E}
    )
    assert bin_.z.dtype == np.float64
    assert bin_.dndz.dtype == np.float64
    np.testing.assert_array_equal(bin_.z, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(bin_.dndz, [0.0, 2.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "args_type", [SourceGalaxyArgs, NumberCountsArgs, WeakLensingArgs]
)
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_direct_source_args(
    args_type: type[SourceGalaxyArgs],
    dtype: type[np.integer[_NBit] | np.floating[_NBit]],
) -> None:
    """Direct construction establishes double storage before systematics.

    :param args_type: Existing public source-argument constructor.
    :param dtype: Supported caller storage representation.
    """
    args = args_type(
        z=np.array([0, 1, 2, 3], dtype=dtype),
        dndz=np.array([0, 2, 1, 0], dtype=dtype),
    )
    assert args.z.dtype == np.float64
    assert args.dndz.dtype == np.float64
    np.testing.assert_array_equal(args.dndz, [0.0, 2.0, 1.0, 0.0])


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_direct_photoz_real_ingress(
    active: bool, dtype: type[np.integer[_NBit] | np.floating[_NBit]]
) -> None:
    """Use a linear distribution with an independently worked weighted mean.

    :param active: Transform the distribution rather than its coordinates.
    :param dtype: Input representation, retained until the public transform.
    """
    transform = (
        dndz_shift_and_stretch_active if active else dndz_shift_and_stretch_passive
    )
    # n(z) = 2z + 1 has weighted mean 70/25 = 2.8 on these five nodes.
    coordinates: npt.NDArray[np.integer[_NBit] | np.floating[_NBit]] = np.array(
        [0, 1, 2, 3, 4], dtype=dtype
    )
    distribution: npt.NDArray[np.integer[_NBit] | np.floating[_NBit]] = np.array(
        [1, 3, 5, 7, 9], dtype=dtype
    )
    z, dndz = transform(
        coordinates,
        distribution,
        0.1,
        0.5,
    )
    assert z.dtype == np.float64
    assert dndz.dtype == np.float64
    np.testing.assert_allclose(
        z,
        [0, 1, 2, 3, 4] if active else [1.3, 1.8, 2.3, 2.8, 3.3],
        rtol=1e-14,
        atol=1e-14,
    )
    if active:
        # The interior spline is exactly linear; padded edge segments need not be.
        np.testing.assert_allclose(dndz[2], 7.6, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(dndz[[0, 1, 4]], [0, 0, 0])
    else:
        np.testing.assert_allclose(dndz, [2, 6, 10, 14, 18], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "values",
    [
        np.array([False, True, True, False]),
        np.array([0, 1 + 2j, 2, 3]),
        np.array(["0", "1", "2", "3"]),
        np.array([0, 1, 2, 3], dtype=object),
        np.zeros(4, dtype=[("value", np.float64)]),
        np.zeros(4, dtype="datetime64[D]"),
        np.ma.array([0.0, 1.0, 2.0, 3.0], mask=False),
        np.ma.array([0.0, 99.0, 2.0, 3.0], mask=[False, True, False, False]),
    ],
)
@pytest.mark.parametrize("field", ["z", "dndz"])
@pytest.mark.parametrize(
    "ingress",
    [
        "tomography",
        "args",
        "counts_args",
        "shear_args",
        "active",
        "passive",
        "counts_read",
        "shear_read",
        "extract",
        "projected",
    ],
)
def test_source_rejects_unsupported_physical_arrays(
    values: npt.NDArray[np.generic], field: str, ingress: str
) -> None:
    """Reject representations before imaginary parts or hidden values are lost.

    :param values: Deliberately unsupported physical-array representation.
    :param field: Coordinate or distribution ingress.
    :param ingress: Existing public boundary under test.
    """
    # These calls deliberately violate static contracts to test runtime rejection.
    z = values if field == "z" else np.array([0.0, 1.0, 2.0, 3.0])
    dndz = values if field == "dndz" else np.array([0.0, 2.0, 1.0, 0.0])
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        if ingress == "tomography":
            TomographicBin(
                bin_name="src",
                z=z,  # type: ignore[arg-type]  # Invalid input.
                dndz=dndz,  # type: ignore[arg-type]  # Invalid input.
                measurements={Galaxies.SHEAR_E},
            )
        elif ingress in {"args", "counts_args", "shear_args"}:
            args_type = {
                "args": SourceGalaxyArgs,
                "counts_args": NumberCountsArgs,
                "shear_args": WeakLensingArgs,
            }[ingress]
            args_type(z=z, dndz=dndz)  # type: ignore[arg-type]
        elif ingress == "active":
            dndz_shift_and_stretch_active(z, dndz, 0.1, 1.0)  # type: ignore[arg-type]
        elif ingress == "passive":
            dndz_shift_and_stretch_passive(z, dndz, 0.1, 1.0)  # type: ignore[arg-type]
        else:
            data = sacc.Sacc()
            data.add_tracer("NZ", "src", [0, 1, 2, 3], [0, 2, 1, 0])
            # SACC's constructor strips masks itself. Place the actual ingress
            # representation on its public tracer fields so Firecrown sees it.
            tracer = data.get_tracer("src")
            tracer.z, tracer.nz = z, dndz
            # Preserve exact support: SACC add_ell_cl casts tags to float.
            for ell_value, cell_value in zip([10], [1.0]):
                data.add_data_point(
                    "galaxy_shear_cl_ee", ("src", "src"), cell_value, ell=ell_value
                )
            if ingress == "extract":
                extract_all_tracers_tomographic_bins(data)
            elif ingress == "projected":
                extract_all_tracers_projected_fields(data)
            elif ingress == "counts_read":
                nc.NumberCounts(sacc_tracer="src").read(data)
            else:
                wl.WeakLensing(sacc_tracer="src").read(data)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("ready", [False, True])
def test_source_preparation_real_ingress(
    dtype: type[np.integer[_NBit] | np.floating[_NBit]],
    ready: bool,
    tools_with_vanilla_cosmology: ModelingTools,
) -> None:
    """Prepare real CCL tracers with independent distributions and amplitudes.

    :param dtype: Physical input dtype, supplied directly to SACC.
    :param ready: Prepare via extracted tomography or the source's SACC read.
    :param tools_with_vanilla_cosmology: Real prepared CCL cosmology.
    """
    data = sacc.Sacc()
    # Read sorts these; ready construction retains the ascending extraction order.
    z = np.array([0, 1, 2, 3, 4, 5] if ready else [5, 4, 0, 2, 1, 3], dtype=dtype)
    nz = np.array([0, 1, 2, 1, 0, 0] if ready else [0, 0, 0, 2, 1, 1], dtype=dtype)
    data.add_tracer("NZ", "src", z, nz)
    # Preserve exact support: SACC add_ell_cl casts tags to float.
    for ell_value, cell_value in zip([10], [1.0]):
        data.add_data_point(
            "galaxy_shear_cl_ee", ("src", "src"), cell_value, ell=ell_value
        )
    assert data.get_tracer("src").z.dtype == dtype

    counts_photoz = nc.PhotoZShift("src", active=False)
    shear_photoz = wl.PhotoZShift("src", active=False)
    alignment = wl.MassDependentLinearAlignmentSystematic("src")
    if ready:
        bin_ = extract_all_tracers_tomographic_bins(data)[0]
        assert bin_.z.dtype == np.float64
        counts = nc.NumberCounts.create_ready(bin_, systematics=[counts_photoz])
        shear = wl.WeakLensing.create_ready(bin_, systematics=[shear_photoz, alignment])
    else:
        counts = nc.NumberCounts(sacc_tracer="src", systematics=[counts_photoz])
        shear = wl.WeakLensing(sacc_tracer="src", systematics=[shear_photoz, alignment])
        counts.read(data)
        shear.read(data)
        assert not np.shares_memory(counts.tracer_args.z, data.get_tracer("src").z)
        assert not np.shares_memory(shear.tracer_args.dndz, data.get_tracer("src").nz)
    np.testing.assert_array_equal(counts.tracer_args.z, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(shear.tracer_args.dndz, [0, 1, 2, 1, 0, 0])
    counts.update(ParamsMap({"src_bias": 1.25, "src_delta_z": -0.25}))
    shear.update(
        ParamsMap(
            {
                "src_delta_z": -0.25,
                "ia_amplitude": 2.5,
                "ia_mass_scaling": 0.5,
                "src_red_fraction": 0.3,
                "src_log10_average_halo_mass": 13.5,
            }
        )
    )
    counts_tracers = counts.get_tracers(tools_with_vanilla_cosmology)
    shear_tracers = shear.get_tracers(tools_with_vanilla_cosmology)
    counts_args = counts.current_tracer_args
    shear_args = shear.current_tracer_args
    assert counts_args is not None and shear_args is not None
    for args in (counts_args, shear_args):
        assert args.z.dtype == np.float64
        assert args.dndz.dtype == np.float64
        np.testing.assert_array_equal(args.z, [0.25, 1.25, 2.25, 3.25, 4.25, 5.25])
        np.testing.assert_array_equal(args.dndz, [0, 1, 2, 1, 0, 0])
    assert counts_args.bias is not None
    assert counts_args.bias.dtype == np.float64
    np.testing.assert_array_equal(counts_args.bias, [1.25] * 6)
    assert shear_args.ia_bias is not None
    assert shear_args.ia_bias[0].dtype == np.float64
    assert shear_args.ia_bias[1].dtype == np.float64
    # Equal mean and pivot masses give A_IA * f_red = 2.5 * 0.3 = 0.75.
    np.testing.assert_allclose(shear_args.ia_bias[1], [0.75] * 6)
    for tracers in (counts_tracers, shear_tracers):
        spectrum = pyccl.angular_cl(
            tools_with_vanilla_cosmology.get_ccl_cosmology(),
            tracers[0].ccl_tracer,
            tracers[0].ccl_tracer,
            np.array([10, 100]),
        )
        assert spectrum.dtype == np.float64
        assert np.all(np.isfinite(spectrum))
        assert np.all(spectrum > 0)


@pytest.mark.parametrize("strided", [False, True])
def test_conforming_source_storage_sharing(strided: bool) -> None:
    """Retain supported sharing without imposing a contiguity policy.

    :param strided: Supply a non-contiguous array view.
    """
    z = np.arange(10, dtype=np.float64)
    if strided:
        z = z[::2]
    bin_ = TomographicBin(bin_name="src", z=z, dndz=z, measurements={Galaxies.SHEAR_E})
    for values in (bin_.z, bin_.dndz, SourceGalaxyArgs(z=z, dndz=z).z):
        assert np.shares_memory(values, z)
    source = wl.WeakLensing.create_ready(bin_)
    assert np.shares_memory(source.tracer_args.z, z)
    assert np.shares_memory(source.tracer_args.dndz, z)


def test_fractional_alignment_from_integer_redshifts(
    tools_with_vanilla_cosmology: ModelingTools,
) -> None:
    """Retain fractional constant alignment with valid integer SACC redshifts.

    :param tools_with_vanilla_cosmology: Real prepared cosmology for source preparation.
    """
    data = sacc.Sacc()
    data.add_tracer(
        "NZ",
        "src",
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        np.array([0, 1, 2, 1, 0, 0], dtype=np.int32),
    )
    source = wl.WeakLensing(
        sacc_tracer="src",
        systematics=[wl.MassDependentLinearAlignmentSystematic("src")],
    )
    source.read(data)
    source.update(
        ParamsMap(
            {
                "ia_amplitude": 2.5,
                "ia_mass_scaling": 0.5,
                "src_red_fraction": 0.3,
                "src_log10_average_halo_mass": 13.5,
            }
        )
    )
    source.get_tracers(tools_with_vanilla_cosmology)
    args = source.current_tracer_args
    assert args is not None and args.ia_bias is not None
    # Equal masses give A_IA * f_red = 0.75 independently of redshift.
    np.testing.assert_allclose(args.ia_bias[1], [0.75] * 6)
    assert args.ia_bias[1].dtype == np.float64


def test_direct_source_bias_ingress() -> None:
    """Tabulated optional bias/alignment arguments also establish double storage."""
    z = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    values = np.array([0, 1, 2, 1, 0], dtype=np.float32)
    bias_z = z.astype(np.float64)
    bias_values = values.astype(np.float64)
    counts = NumberCountsArgs(
        z=z,
        dndz=values,
        bias=bias_values,
        mag_bias=(bias_z, bias_values),
        b_2=(bias_z, bias_values),
        b_s=(bias_z, bias_values),
    )
    shear = WeakLensingArgs(
        z=z,
        dndz=values,
        ia_bias=(bias_z, bias_values),
        ia_pt_c_1=(bias_z, bias_values),
        ia_pt_c_d=(bias_z, bias_values),
        ia_pt_c_2=(bias_z, bias_values),
        ia_a_1h=bias_values,
        ia_a_2h=bias_values,
    )
    arrays = [counts.bias, shear.ia_a_1h, shear.ia_a_2h]
    for pair in (
        counts.mag_bias,
        counts.b_2,
        counts.b_s,
        shear.ia_bias,
        shear.ia_pt_c_1,
        shear.ia_pt_c_d,
        shear.ia_pt_c_2,
    ):
        assert pair is not None
        assert pair[0].dtype == np.float64
        np.testing.assert_array_equal(pair[0], [0, 1, 2, 3, 4])
        arrays.append(pair[1])
    for array in arrays:
        assert array is not None
        assert array.dtype == np.float64
        np.testing.assert_array_equal(array, [0, 1, 2, 1, 0])
