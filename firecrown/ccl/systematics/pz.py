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
    """ A photo-z systematic.

    Convolves the original redshift distribution with a gaussian filter
    with mean mu and std sigma that evolve with redshift according to
    mu_shift = z + mu_0 + (1+z)*mu_1 and sigma_shift = (1+z)*sigma

    TODO: use more efficient integration and test more complex parametrizations

    Parameters:
    -----------
    mu_0 : str
        The name of the first parameter that describes the evolution of the
        mean function
    mu_1 : str
        The name of the second parameter that describes the evolution of the
        mean function
    sigma : str
        The name of the parameter that describes the evolution of the
        std function

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self, mu_0, mu_1, sigma):
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.sigma = sigma

    def apply(self, cosmo, params, source):
        """Apply the systematic to the photo-z distribution of a source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shift.
        """
        _dndz = []
        for z in source.z_:
            gauss_filter = (
                norm.pdf(z, loc=source.z_ + params[self.mu_0] +
                         params[self.mu_1]*(1. + source.z_),
                         scale=params[self.sigma]*(1. + source.z_)))
            joint_distr = gauss_filter*source.dndz_
            _dndz.append(simps(joint_distr, source.z_))
        source.dndz_ = np.array(_dndz)
