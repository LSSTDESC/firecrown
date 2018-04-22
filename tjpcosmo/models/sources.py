class Source:
    def __init__(self, name, stype, metadata):
        self.name = name
        self.stype = stype
        self.systematics = []
        self.metadata = metadata


    def apply_source_systematic(self):
        pass

    def to_tracer(self):
        # return CCL.CLTracer
        pass

    def validate(self):
        pass


class WLSource(Source):
	def __init__(self, name, stype, metadata):
		super(WLSource, self).__init__(self, name, stype, metadata)
		
		
class LSSSource(Source):
	def __init__(self, name, stype, metadata):
		super(LSSSource, self).__init__(self, name, stype, metadata)
		self.z,self.nz = metadata[name]["nz"]
		self.orignal_nz = self.nz
	
	
class SLSource(Source):
	def __init__(self, name, stype, metadata):
		super(SLSource, self).__init__(self, name, stype, metadata)

class Clustersource(Source):
	def __init__(self, name, stype, metadata):
		super(Clustersource, self).__init__(self, name, stype, metadata)

class CMBsource(Source):
	def __init__(self, name, stype, metadata):
		super(CMBsource, self).__init__(self, name, stype, metadata)


def make_source(sname, stype, metadata):
	if stype=='WL':
		return WLSource(...)
	elif stype=='LSS':
		return LSSSource(...)
	elif stype=='SL':
		return SLSource(...)
	elif stype=='CL':
		return Clustersource(...)
	elif stype=='CMB':
		return CMBsource(...)
	elif stype=='Supernova':
		return SNsource(...)
	else:
		raise ValueError(f"The source {stype} asked for doesn't exist in our data!")
