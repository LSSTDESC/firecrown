class Source:
    def __init__(self, name, stype):
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

