import os
import numpy as np
import pytest

from scipy.interpolate import Akima1DInterpolator

import sacc
import pyccl as ccl

from ..parser import parse
from ..loglike import compute_loglike
from ..io import write_statistics


@pytest.fixture(scope="session")
def tx_data(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("data"))

    sacc_data = sacc.Sacc()

    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67,
        transfer_function="eisenstein_hu",
    )

    seed = 42
    rng = np.random.RandomState(seed=seed)
    eps = 0.01

    tracers = []
    for i, mn in enumerate([0.25, 0.75]):
        z = np.linspace(0, 2, 50)
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)

        sacc_data.add_tracer("NZ", "trc%d" % i, z, dndz)

        tracers.append(ccl.WeakLensingTracer(cosmo, dndz=(z, dndz)))

    dv = []
    ndv = []
    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            ell = np.logspace(1, 4, 10)
            pell = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell)
            npell = pell + rng.normal(size=pell.shape[0]) * eps * pell

            sacc_data.add_ell_cl(
                "galaxy_shear_cl_ee", "trc%d" % i, "trc%d" % j, ell, npell
            )
            dv.append(pell)
            ndv.append(npell)

    mn = 0.25
    z = np.linspace(0, 2, 50)
    dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)
    dndz /= np.max(dndz)
    sacc_data.add_tracer(
        "NZ",
        "cl1",
        z,
        dndz,
        metadata={"lnlam_min": np.log(1e14), "lnlam_max": np.log(1e16), "area_sd": 200},
    )

    intp = Akima1DInterpolator(z, dndz)

    def _sel(m, a):
        a = np.atleast_1d(a)
        m = np.atleast_1d(m)
        z = 1.0 / a - 1.0
        logm = np.log10(m)
        zsel = intp(z)
        msk = ~np.isfinite(zsel)
        zsel[msk] = 0.0
        vals = np.zeros((m.shape[0], a.shape[0]))
        vals[:] = zsel
        msk = (logm >= 14) & (logm < 16)
        vals[~msk, :] = 0
        return vals

    mdef = ccl.halos.MassDef(200, "matter")
    hmf = ccl.halos.MassFuncTinker10(cosmo, mdef, mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef, mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(
        cosmo, hmf, hbf, mdef, integration_method_M="spline", nlog10M=256
    )

    true_cnts = hmc.number_counts(cosmo, _sel, amin=0.333333, amax=1, na=256)
    true_cnts *= 200 * (np.pi / 180.0) ** 2
    ntrue_cnts = true_cnts + rng.normal() * eps * true_cnts
    sacc_data.add_data_point(
        "count",
        ("cl1",),
        ntrue_cnts,
    )

    dv.append(np.array([true_cnts]))
    ndv.append(np.array([ntrue_cnts]))

    # a fake covariance matrix
    dv = np.concatenate(dv, axis=0)
    ndv = np.concatenate(ndv, axis=0)
    cov = np.zeros((dv.shape[0], dv.shape[0]))
    for i in range(len(dv)):
        cov[i, i] = (eps * dv[i]) ** 2
    sacc_data.add_covariance(cov)

    sacc_data.save_fits(os.path.join(tmpdir, "sacc.fits"), overwrite=True)

    cinv = np.linalg.inv(cov)
    delta = ndv - dv
    loglike = -0.5 * np.dot(delta, np.dot(cinv, delta))

    config = """\
parameters:
  Omega_k: 0.0
  Omega_c: 0.27
  Omega_b: 0.045
  h: 0.67
  n_s: [0.9, 0.96, 1.0]
  sigma8: 0.8
  w0: -1.0
  wa: 0.0
  transfer_function: 'eisenstein_hu'

  # lens bin zero
  src0_delta_z: [-0.1, 0.0, 0.1]
  src1_delta_z: 0.0

two_point_plus_clusters:
  module: firecrown.ccl
  sacc_data: {tmpdir}/sacc.fits
  sources:
    src0:
      kind: WLSource
      sacc_tracer: trc0
      systematics:
        - pz_delta_0

    src1:
      kind: WLSource
      sacc_tracer: trc1
      systematics:
        - pz_delta_1

    src2:
      kind: ClusterSource
      sacc_tracer: cl1

  systematics:
    pz_delta_0:
      kind: PhotoZShiftBias
      delta_z: src0_delta_z

    pz_delta_1:
      kind: PhotoZShiftBias
      delta_z: src1_delta_z

  likelihood:
    kind: ConstGaussianLogLike
    data_vector:
      - cl_src0_src0
      - cl_src0_src1
      - cl_src1_src1
      - counts_src2

  statistics:
    cl_src0_src0:
      kind: TwoPointStatistic
      sources: ['src0', 'src0']
      sacc_data_type: galaxy_shear_cl_ee

    cl_src0_src1:
      kind: TwoPointStatistic
      sources: ['src0', 'src1']
      sacc_data_type: galaxy_shear_cl_ee

    cl_src1_src1:
      kind: TwoPointStatistic
      sources: ['src1', 'src1']
      sacc_data_type: galaxy_shear_cl_ee

    counts_src2:
      kind: ClusterCountStatistic
      sources: ['src2']
      mass_def: [200, 'matter']
      mass_func: Tinker10
      halo_bias: Tinker10
      na: 256
      nlog10M: 256
""".format(
        tmpdir=tmpdir
    )

    with open(os.path.join(tmpdir, "config.yaml"), "w") as fp:
        fp.write(config)

    return {
        "cosmo": cosmo,
        "tmpdir": tmpdir,
        "loglike": loglike,
        "config": config,
        "cov": cov,
        "inv_cov": cinv,
    }


def test_integration_generic_ccl_smoke(tx_data):
    tmpdir = tx_data["tmpdir"]
    cfg_path = os.path.join(tmpdir, "config.yaml")

    config, data = parse(cfg_path)
    loglike, meas, pred, covs, inv_covs, stats = compute_loglike(
        cosmo=tx_data["cosmo"], data=data
    )

    assert np.allclose(loglike["two_point_plus_clusters"], tx_data["loglike"])

    write_statistics(
        output_dir=os.path.join(tmpdir, "output_123"),
        data=data,
        statistics=stats,
    )

    opth = os.path.join(tmpdir, "output_123", "statistics", "two_point_plus_clusters")
    orig_data = sacc.Sacc.load_fits(os.path.join(tmpdir, "sacc.fits"))
    meas_data = sacc.Sacc.load_fits(os.path.join(opth, "sacc_measured.fits"))
    pred_data = sacc.Sacc.load_fits(os.path.join(opth, "sacc_predicted.fits"))

    for trc_name in ["trc0", "trc1", "cl1"]:
        orig_tr = orig_data.get_tracer(trc_name)
        meas_tr = meas_data.get_tracer(trc_name)
        pred_tr = pred_data.get_tracer(trc_name)

        assert np.allclose(orig_tr.z, meas_tr.z)
        assert np.allclose(orig_tr.z, pred_tr.z)
        assert np.allclose(orig_tr.nz, meas_tr.nz)
        assert np.allclose(orig_tr.nz, pred_tr.nz)

    meas_dv = []
    pred_dv = []
    for trs in [("trc0", "trc0"), ("trc0", "trc1"), ("trc1", "trc1")]:
        oell, ocl = orig_data.get_ell_cl("galaxy_shear_cl_ee", trs[0], trs[1])
        mell, mcl = meas_data.get_ell_cl("galaxy_shear_cl_ee", trs[0], trs[1])
        pell, pcl = pred_data.get_ell_cl("galaxy_shear_cl_ee", trs[0], trs[1])

        assert np.allclose(oell, mell)
        assert np.allclose(oell, pell)
        assert np.array_equal(ocl, mcl)
        assert not np.array_equal(ocl, pcl)
        meas_dv.append(mcl)
        pred_dv.append(pcl)

    meas_dv.append(np.atleast_1d(meas_data.get_data_points("count", ("cl1",))[0].value))
    pred_dv.append(np.atleast_1d(pred_data.get_data_points("count", ("cl1",))[0].value))

    assert np.allclose(np.concatenate(meas_dv, axis=0), meas["two_point_plus_clusters"])
    assert np.allclose(np.concatenate(pred_dv, axis=0), pred["two_point_plus_clusters"])
    assert np.allclose(covs["two_point_plus_clusters"], tx_data["cov"])
    assert np.allclose(inv_covs["two_point_plus_clusters"], tx_data["inv_cov"])
