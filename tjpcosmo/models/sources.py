class Source:
    def __init__(self, name, stype, metadata):
        self.name = name
        self.stype = stype
        self.systematics = []


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
	if sname=='WL':
		return WLSource(...)
	elif sname=='LSS'
		return LSSSource(...)
	elif sname=='SL':
		return SLSource(...)
	elif sname=='Cluster':
		return Clustersource(...)
	elif sname=='CMB':
		return CMBsource(...)
	elif sname=='Supernova':
		return SNsource(...)
	else 
		raise ValueError(f"The source {sname} asked for doesn't exist in our data!")
