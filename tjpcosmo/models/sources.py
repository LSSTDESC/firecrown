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
        # return CCL.CLTracer
        pass

    def validate(self):
        pass


class WLSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(self, name, stype, metadata)
        self.scaling = 1.0
        
        
class LSSSource(Source):
    def __init__(self, name, stype, metadata):
	super().__init__(name, stype, metadata)
	self.z,self.nz = metadata['sources'][name]["nz"]
	self.orignal_nz = self.nz
        
class SLSource(Source):
    def __init__(self, name, stype, metadata):
        super(SLSource, self).__init__(   name, stype, metadata)

class SNSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(self, name, stype, metadata)

class CLSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(self, name, stype, metadata)

class CMBSource(Source):
    def __init__(self, name, stype, metadata):
        super().__init__(self, name, stype, metadata)


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
