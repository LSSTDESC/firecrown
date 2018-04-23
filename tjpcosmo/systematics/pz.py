import numpy as np
from scipy.interpolate import Akima1DInterpolator
from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic

class PZShift(SourceSystematic):
    modified_source_properties =['nz']
    required_source_properties =['z']
    params = ['delta_z']
    def adjust_source(self, cosmo, source):
        if self.adjust_requirements(source):
            source.nz = source.nz_interp(source.z-self.values['delta_z'], extrapolate = False)
            source.nz[np.isnan(source.nz)] = 0.0
            return 0
        else:
            print(f"{self.__class__.__name__} did not find all required source parameters")
            return 1

