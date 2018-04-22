from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic


class MultiplicativeShearBias(SourceSystematic):
    params = m
    def adjust_source(self, cosmo, source):
        source.scaling[:] = self.scaling[:]*m    


class PZTransformation(SourceSystematic):
    pass

class PZShift(PZTransformation):
    pass

class AdditiveShearBias(OutputSystematic):
    pass

class LinearAlignment(SourceSystematic):
    pass

class BaryonEffects(CosmologySystematic):
    pass
