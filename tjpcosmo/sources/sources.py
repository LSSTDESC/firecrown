import numpy as np
import copy
from scipy.interpolate import Akima1DInterpolator

class Source:
    """
    The super class for Sources, any source should be a made as a subclass of this.
    A source need to know how to initialize it self and how to it can get transformed
    into a tracer, and how it resets itself to the its original values and how to
    validate it self. Validation is however not implemented yet! blane Joe Zuntz
    
    Each source also have a scaling factor, it should be 1, but certain effects might
    effect this. subclasses and systematics might affect this, but if nothing else is 
    given it is scaled to 1.
    """
    def __init__(self, name, stype, metadata):
        self.name = name
        self.stype = stype
        self.systematics = []
        self.eval_source_prop=['z']
        self.metadata = metadata
        self.scaling = 1.0
        self.sys_order = None
        
        """
        Args:
            name(String) the name of the source
            stype(string) source type
            metadata(object) metadata, should be loaded automatically.
        """

    @classmethod
    def from_dict(cls, name, info):

        stype = info['type']
        cls(name, stype, )


    def to_tracer(self):
        raise ValueError("Wrong kind of source turned into a tracer!")

    def validate(self):
        pass

    def reset(self):
        raise NotImplementedError(f"Need to implement reset method for source subclass {self.__class__.__name__}")

    def apply_systematics(self, cosmo):
        if self.sys_order is None:
            # This first call which figures out the ordering
            # also as a by-product applies the systematics
            self.order_systematics(cosmo)
        else:
            for sys in self.sys_order:
                err = sys.adjust_source(cosmo, self)
                if err:
                    raise ValueError(f"Non-deterministic behaviour in source {self.name} systematic {sys.__class__.__name__} ordering found.")

    def order_systematics(self, cosmo):
        print(f"Determining order of application of systematics for source: {self.name}")
        self.sys_order = []
        self.reset()
        sysloop=self.systematics
        while sysloop:
            missing_systematics=[]
            prevlength=len(sysloop)
            for sys in self.systematics:
                if sys.adjust_source(cosmo, self):
                    missing_systematics.append(sys)
                else:
                    self.sys_order.append(sys)
                sysloop=missing_systematics.copy()
                if sysloop:
                    print(f"Encountered nested systematics, entering next iteration")
                if len(sysloop)==prevlength:
                    raise ValueError(f"Could not reduce size of systematics list: NESTED")


class WLSource(Source):
    """ Weak lensing Source.
        Expected data:
            z
            nz
    
        Systematics:
            f_red
            ia.amplitude
            
    """
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)
        self.z = metadata['sources'][name]["z"]
        self.original_nz = metadata['sources'][name]["nz"]
        self.nz_interp = Akima1DInterpolator(self.z, self.original_nz)
        print("Are we calling reset in the wrong places?")
        self.reset()

    def reset(self):
        self.f_red = np.ones_like(self.z)
        self.ia_amplitude = np.ones_like(self.z)
        self.nz = self.original_nz.copy()
        self.scaling = 1.0
        
    def to_tracer(self, cosmo):
        import pyccl as ccl
        if(np.any(self.ia_amplitude!=0) & np.any(self.f_red!=0)):
            tracer = ccl.ClTracerLensing(cosmo, has_intrinsic_alignment=True,
                     n=(self.z,self.nz), bias_ia=(self.z, self.ia_amplitude),
                     f_red=(self.z,self.f_red))
        else:
            tracer = ccl.ClTracerLensing(cosmo, has_intrinsic_alignment=False,
                     n=(self.z,self.nz))
        return tracer


class LSSSource(Source):
    """ Large scale structure source:
        Expected data:
            z
            nz
            
        Systematics:
            
    """
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)
        self.z = metadata['sources'][name]["z"]
        self.original_nz = metadata['sources'][name]["nz"]
        self.nz_interp = Akima1DInterpolator(self.z, self.original_nz)
        self.reset()

    def reset(self):
        self.bias = np.ones_like(self.z)
        self.nz = self.original_nz.copy()
        self.scaling = 1.0


    def to_tracer(self, cosmo):
        import pyccl as ccl
        print("Put RSD and mag in here when needed")
        tracer = ccl.ClTracerNumberCounts(cosmo, has_rsd=False, has_magnification=False, 
            n=(self.z,self.nz), bias=(self.z, self.bias))
        return tracer

class SLSource(Source):
    """ Strong Lensing source:
        Expected data:
            
        Systematics:
    """
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)

class SNSource(Source):
    """ SuperNovae sourec:
        Expected data:
            
        Systematics:
    """
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)

class CLSource(Source):
    """ Cluster source:
        Expected data:
            
        Systematics:
            
    """
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)

class CMBSource(Source):
    """ Cosmic Microwave background Source:
        Expected data:
            
        Systematics:
            
    """
    def __init__(self, name, stype, metadata):
        super().__init__(name, stype, metadata)


def make_source(sname, stype, metadata):
    """ Makes a source from the input in the config .yaml file, assigning it to
    the right type of source. or tells you that the type of source isn't implemented.
    
    Args:
        Sname(string): name of the source
        stype(string): type of the source
        metadata(object): created automatically from the config file. 
    """
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