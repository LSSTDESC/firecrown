import os
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate

import pyccl as ccl

N_BINS = 5
LENS_KMAX = 0.3 * 0.6727  # little h sucks
SRC_LMAX = 3e3
ELL_VALUES = np.array([
    2.359970e+01,
    3.285940e+01,
    4.575227e+01,
    6.370386e+01,
    8.869901e+01,
    1.235014e+02,
    1.719590e+02,
    2.394297e+02,
    3.333735e+02,
    4.641775e+02,
    6.463045e+02,
    8.998917e+02,
    1.252978e+03,
    1.744602e+03,
    2.429122e+03,
    3.382225e+03,
    4.709291e+03,
    6.557052e+03,
    9.129808e+03,
    1.271202e+04])

# the fiducial cosmology from the SRD
COSMO = ccl.Cosmology(
    Omega_b=0.0492,
    Omega_c=0.26639999999999997,  # = 0.3156 - 0.0492
    w0=-1.0,
    wa=0.0,
    h=0.6727,
    A_s=2.12655e-9,  # has sigma8 = 0.8310036
    n_s=0.9645)


def make_lens_src_ell_bins(output_dir, mean_z):
    # here we have to combine the small-scale cuts and the linear bias cuts
    for lens_i in range(N_BINS):
        for src_j in range(N_BINS):
            lmax_lens = (
                LENS_KMAX * ccl.comoving_radial_distance(
                    COSMO, 1.0 / (1.0 + mean_z[lens_i])) - 0.5)
            lmax = min(lmax_lens, SRC_LMAX)
            msk = ELL_VALUES < lmax
            df = pd.DataFrame({'ell_or_theta': ELL_VALUES[msk]})
            df.to_csv(
                os.path.join(
                    output_dir, 'ell_lens%d_src%d.csv' % (lens_i, src_j)),
                index=False)


def make_lens_ell_bins(output_dir, mean_z):
    # these cuts make sure we are in the linear bias regime for the
    # given redshift distribution
    for i in range(N_BINS):
        lmax = (
            LENS_KMAX * ccl.comoving_radial_distance(
                COSMO, 1.0 / (1.0 + mean_z[i])) - 0.5)
        msk = ELL_VALUES < lmax
        df = pd.DataFrame({'ell_or_theta': ELL_VALUES[msk]})
        df.to_csv(
            os.path.join(output_dir, 'ell_lens%d_lens%d.csv' % (i, i)),
            index=False)


def make_src_ell_bins(output_dir):
    # we cut the fiducial binning below SRC_LMAX to account for baryons
    # and general small-scale problems
    msk = ELL_VALUES < SRC_LMAX
    df = pd.DataFrame({'ell_or_theta': ELL_VALUES[msk]})
    for i in range(N_BINS):
        for j in range(i, N_BINS):
            df.to_csv(
                os.path.join(output_dir, 'ell_src%d_src%d.csv' % (i, j)),
                index=False)


def make_lens_z_bins(output_dir):
    zmin = 0.2
    zmax = 1.2
    dz = (zmax - zmin) / N_BINS

    for i in range(N_BINS):
        _zmin = i * dz + zmin
        _zmax = _zmin + dz

        z, dndz = _make_pz(_zmin, _zmax, _pz_lens, _sigmaz_lens)

        msk = dndz > 0
        df = pd.DataFrame({'z': z[msk], 'dndz': dndz[msk]})
        df.to_csv(os.path.join(output_dir, 'lens%d_dndz.csv' % i), index=False)


def _pz_lens(z, z0=0.26, alpha=0.94):
    return z * z * np.exp(-np.power(z/z0, alpha))


def _sigmaz_lens(z):
    return 0.03 * (1.0 + z)


def make_src_z_bins(output_dir):
    # we are making equal number density bins here
    # idea is to invert the cumulative dndz and then find the
    # redshifts that divide the distribution equally
    zarr = np.linspace(0, 5.0, 10000)
    dndz_true = _pz_src(zarr)
    cuml_dndz_true = np.cumsum(dndz_true / np.sum(dndz_true))
    interp = scipy.interpolate.interp1d(
        cuml_dndz_true, zarr,
        fill_value='extrapolate', kind='cubic')
    z_cutoffs = interp(np.linspace(0, 1, N_BINS+1))
    z_cutoffs[-1] = np.inf

    # the SRD convolves the photoz scatter into the "true" distribution in
    # order to define the redshift distribution of each bin. As stated there,
    # this is NOT correct but easier to implememt.
    for i in range(N_BINS):
        _zmin = z_cutoffs[i]
        _zmax = z_cutoffs[i+1]

        z, dndz = _make_pz(_zmin, _zmax, _pz_src, _sigmaz_src)

        msk = dndz > 0
        df = pd.DataFrame({'z': z[msk], 'dndz': dndz[msk]})
        df.to_csv(os.path.join(output_dir, 'src%d_dndz.csv' % i), index=False)


def _pz_src(z, z0=0.13, alpha=0.78):
    return z * z * np.exp(-np.power(z/z0, alpha))


def _sigmaz_src(z):
    return 0.05 * (1.0 + z)


def _make_pz(z_true_min, z_true_max, _pz, _sigmaz, n_z=1000):
    """Makes a p(z) by convolving some scatter into a "true" redshift
    distribution in some range (z_true_min, z_true_max).
    """
    z_obs = np.linspace(0.0, 4.0, n_z)
    dz_obs = z_obs[1] - z_obs[0]
    z_obs += dz_obs
    dndz = np.zeros_like(z_obs)

    for i, _z in enumerate(z_obs):

        def _func(z):
            return (
                _pz(z) *
                np.exp(-0.5 * np.power((z - _z) / _sigmaz(z), 2)) /
                _sigmaz(z))

        dndz[i] = scipy.integrate.quad(_func, z_true_min, z_true_max)[0]

    nrm = np.trapz(y=dndz, x=z_obs)
    return z_obs, dndz / nrm
