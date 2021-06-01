import numpy as np
import scipy.interpolate

KCORR_DATA = """\
0.02 r  0.021  0.020  0.015  0.008
0.04 r  0.050  0.050  0.036  0.020
0.06 r  0.065  0.064  0.046  0.023
0.08 r  0.079  0.078  0.056  0.026
0.10 r  0.102  0.101  0.073  0.035
0.12 r  0.123  0.123  0.090  0.044
0.16 r  0.171  0.170  0.129  0.065
0.20 r  0.246  0.245  0.190  0.104
0.24 r  0.300  0.299  0.234  0.128
0.28 r  0.361  0.359  0.283  0.157
0.32 r  0.416  0.414  0.329  0.181
0.36 r  0.470  0.468  0.373  0.202
0.40 r  0.541  0.538  0.432  0.234
0.44 r  0.651  0.647  0.521  0.285
0.48 r  0.747  0.744  0.600  0.328
0.52 r  0.855  0.851  0.688  0.376
0.60 r  1.148  1.143  0.928  0.519
0.68 r  1.455  1.449  1.186  0.701
0.76 r  1.763  1.754  1.442  0.890
0.84 r  1.936  1.926  1.592  1.010
0.92 r  2.159  2.148  1.765  1.107
1.00 r  2.294  2.282  1.873  1.167
1.10 r  2.523  2.510  2.032  1.219
1.20 r  2.844  2.830  2.241  1.278
1.30 r  3.165  3.148  2.427  1.317
1.40 r  3.496  3.478  2.598  1.343
1.50 r  3.950  3.927  2.784  1.365
1.60 r  4.308  4.280  2.901  1.365
1.70 r  4.781  4.744  3.006  1.349
1.80 r  4.776  4.734  2.953  1.284
1.90 r  5.133  5.079  2.980  1.242
2.00 r  5.319  0.253  2.967  1.196
2.10 r  5.473  5.395  2.951  1.154
2.20 r  5.709  5.601  2.949  1.123
2.30 r  5.857  5.743  2.933  1.092
2.40 r  5.992  5.858  2.909  1.054
2.50 r  6.182  6.019  2.887  1.018
3.00 r  6.888  6.558  2.804  0.897"""

KCORR_Z = np.array([float(line.strip().split()[0]) for line in KCORR_DATA.split("\n")])
KCORR_VAL = np.array(
    [float(line.strip().split()[2]) for line in KCORR_DATA.split("\n")]
)
KCORR_SPLINE = scipy.interpolate.interp1d(
    KCORR_Z,
    KCORR_VAL,
    kind="cubic",
    bounds_error=False,
    fill_value=(KCORR_VAL[0], KCORR_VAL[-1]),
)

ECORR_DATA = """\
0.02 r -0.024 -0.026 -0.035 -0.028
0.04 r -0.048 -0.053 -0.071 -0.055
0.06 r -0.073 -0.077 -0.105 -0.082
0.08 r -0.097 -0.103 -0.140 -0.109
0.10 r -0.120 -0.127 -0.175 -0.135
0.12 r -0.142 -0.151 -0.210 -0.161
0.16 r -0.187 -0.199 -0.283 -0.212
0.20 r -0.233 -0.250 -0.355 -0.263
0.24 r -0.277 -0.300 -0.429 -0.312
0.28 r -0.322 -0.351 -0.505 -0.361
0.32 r -0.365 -0.401 -0.581 -0.409
0.36 r -0.408 -0.453 -0.660 -0.457
0.40 r -0.450 -0.505 -0.743 -0.507
0.44 r -0.496 -0.564 -0.836 -0.563
0.48 r -0.541 -0.629 -0.932 -0.617
0.52 r -0.589 -0.696 -1.032 -0.672
0.60 r -0.699 -0.861 -1.260 -0.793
0.68 r -0.822 -1.039 -1.466 -0.895
0.76 r -0.960 -1.232 -1.658 -0.983
0.84 r -1.085 -1.397 -1.800 -1.048
0.92 r -1.236 -1.608 -1.990 -1.132
1.00 r -1.378 -1.791 -2.141 -1.198
1.10 r -1.613 -2.095 -2.389 -1.301
1.20 r -1.912 -2.476 -2.681 -1.416
1.30 r -2.264 -2.890 -2.964 -1.520
1.40 r -2.661 -3.334 -3.236 -1.615
1.50 r -3.215 -3.912 -3.527 -1.684
1.60 r -3.714 -4.406 -3.749 -1.775
1.70 r -4.354 -5.025 -3.970 -1.839
1.80 r -4.597 -5.216 -4.055 -1.870
1.90 r -5.158 -5.732 -4.207 -1.915
2.00 r -5.545 -6.069 -4.312 -1.948
2.10 r -5.896 -6.368 -4.410 -1.980
2.20 r -6.300 -6.718 -4.504 -2.010
2.30 r -6.611 -6.978 -4.578 -2.031
2.40 r -6.905 -7.218 -4.645 -2.046
2.50 r -7.243 -7.496 -4.709 -2.056
3.00 r -8.549 -8.499 -4.961 -2.054"""

ECORR_Z = np.array([float(line.strip().split()[0]) for line in ECORR_DATA.split("\n")])
ECORR_VAL = np.array(
    [float(line.strip().split()[2]) for line in ECORR_DATA.split("\n")]
)
ECORR_SPLINE = scipy.interpolate.interp1d(
    ECORR_Z,
    ECORR_VAL,
    kind="cubic",
    bounds_error=False,
    fill_value=(ECORR_VAL[0], ECORR_VAL[-1]),
)


def kcorr(z):
    """Return the k-correction for an E-type galaxy according to the models
    in Poggianti B. M., 1997, A&AS, 122, 399.

    A k-correction is the correction to the magntiude of a galaxy at a given
    time and effective wavelength for cosmological redshifting between the
    time the light was emitted and today.

    Parameters
    ----------
    z : float or array-like
        The redshift at which to evaluate the k-correction.

    Returns
    -------
    kcorr : float or array-like
        The k-correction.
    """
    return KCORR_SPLINE(z)


def ecorr(z):
    """Return the e-correction for an E-type galaxy according to the models
    in Poggianti B. M., 1997, A&AS, 122, 399.

    An e-correction is the change in the magnitude of a galaxy just due to
    the intrinsic evolution of the objects spectrum.

    Parameters
    ----------
    z : float or array-like
        The redshift at which to evaluate the e-correction.

    Returns
    -------
    ecorr : float or array-like
        The e-correction.
    """
    return ECORR_SPLINE(z)
