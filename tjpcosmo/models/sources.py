import numpy as np
import copy

class Source:
    def __init__(self, name, stype, metadata):
        self.name = name
        self.stype = stype
        self.systematics = []
        self.metadata = metadata


    def apply_source_systematic(self):
        for sys in self.systematics:
            sys.apply(self)

    def to_tracer(self):
        raise ValueError("Wrong kind of source turned into a tracer!")

    def validate(self):
        pass
    def copy(self):
        return copy.copy(self)



class WLSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)
        self.z,self.nz = metadata['sources'][name]["nz"]
        self.orignal_nz = self.nz
        self.f_red = np.ones_like(self.z)
        self.ia_amplitude = np.ones_like(self.z)
        #self.scaling = 1.0
        
    def to_tracer(self, cosmo):
        import pyccl as ccl
        if(np.any(self.ia_amplitude!=0) & np.any(self.f_red!=0)):
            print("with alignment\n\n\n")
            tracer = ccl.ClTracerLensing(cosmo, has_intrinsic_alignment=True,
                     n=(self.z,self.nz), bias_ia=(self.z, self.ia_amplitude),
                     f_red=(self.z,self.f_red))
        else:
            print("no alignment\n\n\n")
            tracer = ccl.ClTracerLensing(cosmo, has_intrinsic_alignment=False,
                     n=(self.z,self.nz))
        return tracer


class LSSSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)
        self.z,self.nz = metadata['sources'][name]["nz"]
        self.orignal_nz = self.nz
        self.bias = np.ones_like(self.z)

    def to_tracer(self, cosmo):
        import pyccl as ccl
        print("Put RSD and mag in here when needed")
        tracer = ccl.ClTracerNumberCounts(cosmo, has_rsd=False, has_magnification=False, 
            n=(self.z,self.nz), bias=(self.z, self.bias))
        return tracer

class SLSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)

class SNSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)

class CLSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)

class CMBSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)


def make_source(sname, stype, metadata):
    if stype=='WL':
        return WLSource(sname, stype, metadata)
    elif stype=='LSS':
        return LSSSource(sname, stype, metadata)
    elif stype=='SL':
        return SLSource(sname, stype, metadata)
    elif stype=='CL':
        return Clustersource(sname, stype, metadata)
    elif stype=='CMB':
        return CMBsource(sname, stype, metadata)
    elif stype=='SN':
        return SNsource(sname, stype, metadata)
    else:
        raise ValueError(f"The source {stype} asked for doesn't exist in our data!")
