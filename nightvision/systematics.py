import numpy as np


def photoz_shift(z, spline, delta_z):
    """Apply a photo-z shift to an input photo-z distribution.

    Parameters
    ----------
    z : array-like, shape (n_bins,)
        The redshift values.
    spline : function
        A function that interpolates the photo-z distribution given an input
        redshift value. It should return `np.nan` if the redshift value is out
        of range. These values get replaced with zero.
    delta_z : float
        The photo-z shift.

    Returns
    -------
    n : array-like, shape (n_bins,)
        The shifted photo-z distribution.
    """
    n = spline(z - delta_z)
    n[np.isnan(n)] = 0.0
    return n
