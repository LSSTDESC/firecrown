from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic


class MultiplicativeShearBias(SourceSystematic):
    pass


class PZTransformation(SourceSystematic):
    pass

class PZShift(PZTransformation):
    pass

class AdditiveShearBias(OutputSystematic):
    pass

class LinearAlignment(SourceSystematic):
    params = ['biasia','fred']
    def adjust_source(self,cosmo,source):
        source.ia_amplitude[:]=self.values['biasia']
        source.f_red[:]=self.values['fred']

class BaryonEffects(CosmologySystematic):
    pass
