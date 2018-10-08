import numpy as np
from ..core import SystematicMixin

__all__ = ['PhotoZShiftBias']


class PhotoZShiftBias(SystematicMixin):
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
        _nz = source.nz_interp(
                source.z_ - params[self.delta_z], extrapolate=False)
        _nz[np.isnan(_nz)] = 0.0
        source.nz_ = _nz
