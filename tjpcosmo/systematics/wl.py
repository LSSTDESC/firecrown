from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic


class MultiplicativeShearBias(SourceSystematic):
    params = ['m']
    def adjust_source(self, cosmo, source):
        source.scaling *= (1+ self.values['m'])   



class AdditiveShearBias(OutputSystematic):
    pass

class LinearAlignment(SourceSystematic):
    params = ['biasia','fred']
    def adjust_source(self,cosmo,source):
        source.ia_amplitude[:]=self.values['biasia']
        source.f_red[:]=self.values['fred']

class BaryonEffects(CosmologySystematic):
    pass
