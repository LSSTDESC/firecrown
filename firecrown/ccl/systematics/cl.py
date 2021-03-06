import numpy as np

from ..core import Systematic

__all__ = [
    'IdentityFunctionMOR',
    'TopHatSelectionFunction',
    'PowerLawMOR',
]


class IdentityFunctionMOR(Systematic):
    """An identity function mass-observable relationship.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self):
        pass

    def apply(self, cosmo, params, source):
        """Apply this MOR to the source.

        This method attaches functions `mor_` and `inv_mor_` to be used
        by later functions for compute the MOR and its inverse.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the MOR relationship.
        """
        source.mor_ = self._gen_mor()
        source.inv_mor_ = self._gen_mor()

    def _gen_mor(self):
        def _mor(lnmass, a):
            return lnmass

        return _mor


class PowerLawMOR(Systematic):
    """A power-law mass-observable relationship.

    It has the form

        lam = lam_norm * (m / m_norm)**mass_slope * (a / a_norm)**a_slope

    Note that this function returns lnlam, not lam.

    Parameters
    ----------
    lnlam_norm : str
        The name of the normalization parameter.
    mass_slope : str
        The name of the mass slope parameter.
    a_slope : str
        The name of the scale factor slope parameter.
    lnmass_norm : float
        The natural logarithm of the pivot mass. Default is `np.log(1e14)`.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self, *, lnlam_norm, mass_slope, a_slope, lnmass_norm=np.log(1e14)):
        self.lnlam_norm = lnlam_norm
        self.mass_slope = mass_slope
        self.a_slope = a_slope
        self.lnmass_norm = lnmass_norm

    def apply(self, cosmo, params, source):
        """Apply this MOR to the source.

        This method attaches functions `mor_` and `inv_mor_` to be used
        by later functions for compute the MOR and its inverse.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the MOR relationship.
        """
        source.mor_ = self._gen_mor(params)
        source.inv_mor_ = self._gen_inv_mor(params)

    def _gen_mor(self, params):
        def _mor(lnmass, a):
            return (
                params[self.lnlam_norm]
                + params[self.mass_slope] * (lnmass - self.lnmass_norm)
                + params[self.a_slope] * np.log(a)
            )
        return _mor

    def _gen_inv_mor(self, params):
        def _inv_mor(lnlam, a):
            return (
                lnlam
                - params[self.lnlam_norm]
                - params[self.a_slope] * np.log(a)
            ) / params[self.mass_slope] + self.lnmass_norm
        return _inv_mor


class TopHatSelectionFunction(Systematic):
    """Top-hat selection function for clusters.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self):
        pass

    def apply(self, cosmo, params, source):
        """Apply this selection function to the source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the top-hat selection function.
        """
        source.selfunc_ = self._gen_selection_function(source)

    def _gen_selection_function(self, source):

        def _selfunc(lnmass, a):
            a = np.atleast_1d(a)
            lnmass = np.atleast_1d(lnmass)
            if a.ndim == 1:
                a = a.reshape((1, -1))
            if lnmass.ndim == 1:
                lnmass = lnmass.reshape((-1, 1))
            out_shape = np.broadcast(lnmass, a)
            a = np.broadcast_to(a, out_shape.shape)
            lnmass = np.broadcast_to(lnmass, out_shape.shape)
            vals = np.zeros(out_shape.shape)

            lnmass_min = source.inv_mor_(source.lnlam_min_, a)
            lnmass_max = source.inv_mor_(source.lnlam_max_, a)
            msk = lnmass_max < lnmass_min
            if np.any(msk):
                _tmp = lnmass_min.copy()
                lnmass_min[msk] = lnmass_max[msk]
                lnmass_max[msk] = _tmp[msk]

            msk = (lnmass_min <= lnmass) & (lnmass <= lnmass_max)

            if np.any(msk):
                vals[msk] = source.dndz_interp_(1/a[msk]-1)

            return vals

        return _selfunc
