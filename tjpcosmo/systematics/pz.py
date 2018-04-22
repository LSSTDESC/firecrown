import numpy as np
from scipy.interpolate import Akima1DInterpolator
from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic

class PZShift(SourceSystematic):
    params = ['delta_z']
    def adjust_source(self, cosmo, source):
        nz_interp = Akima1DInterpolator(source.z,source.orignal_nz)
        source.nz = nz_interp(source.z-self.values['delta_z'], extrapolate = False)
        source.nz[np.isnan(source.nz)] = 0.0

