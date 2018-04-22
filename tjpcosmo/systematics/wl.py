from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic


class MultiplicativeShearBias(SourceSystematic):
    params = ['m']
    def adjust_source(self, cosmo, source):
        source.scaling *= (1+ self.values['m'])   



class AdditiveShearBias(OutputSystematic):
    pass

class LinearAlignment(SourceSystematic):
    params = ['biasia']
    optional_params = {
        'alpha': 0.0,
        'z_piv': 0.0,
        'fred': 1.0
    }
    def adjust_source(self,cosmo,source):
        source.ia_amplitude[:]=(((1.+source.z)/(1.+self.values['z_piv']))**(self.values['alpha'])
        *self.values['biasia'])
        source.f_red[:]=self.values['fred']

class BaryonEffects(CosmologySystematic):
    pass
