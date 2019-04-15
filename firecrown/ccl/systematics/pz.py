import numpy as np
from scipy.stats import norm
from scipy.integrate import simps
from ..core import Systematic

__all__ = ['PhotoZShiftBias', 'PhotoZSystematic']


class PhotoZShiftBias(Systematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.

    Parameters
    ----------
    delta_z : str
        The name of the photo-z shift parameter.

    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self, delta_z):
        self.delta_z = delta_z

    def apply(self, cosmo, params, source):
        """Apply a shift to the photo-z distribution of a source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shift.
        """
        _dndz = source.dndz_interp(
                source.z_ - params[self.delta_z], extrapolate=False)
        _dndz[np.isnan(_dndz)] = 0.0
        source.dndz_ = _dndz


class PhotoZSystematic(Systematic):
    """ Convolves the original redshift distribution with a gaussian filter
    with mean mu and std sigma that evolve with redshift
    according to mu_shift = z + para_mean[0] + (1+z)*para_mean[1] and
    sigma_shift = (1+z)*para_std

    TODO: use more efficient integration and test more complex parametrizations

    Parameters:
    -----------
    zorig: midpoints of original histogram bins
    zpdf_orig: histogram heights
    para_mean: parameters that describes the evolution of the mean function
    para_std: parameter that describes the evolution of the std function

    Returns:
    --------
    mod_pdf: Modified histogram bins

    """
    def __init__(self, mu_0, mu_1, sigma):
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.sigma = sigma

    def apply(self, cosmo, params, source):
        zorig, N = (source.z_, len(source.z_))
        zpdf_orig = source.dndz_/np.sum(source.dndz_*source.z_)
        joint_distr = np.zeros((N, N))
        _dndz = np.zeros(N)

        for i in range(N):
            gauss_filter = (
                norm.pdf(zorig[i], loc=zorig + params[self.mu_0] +
                         params[self.mu_1]*(1. + zorig),
                         scale=params[self.sigma]*(1. + zorig)))
            joint_distr[i, :] = gauss_filter*zpdf_orig
            _dndz[i] = simps(joint_distr[i, :], zorig)

        source.dndz_ = _dndz/simps(_dndz, zorig)
