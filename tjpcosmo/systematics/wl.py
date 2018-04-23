from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic
import pyccl as ccl


class MultiplicativeShearBias(SourceSystematic):
    params = ['m']
    def adjust_source(self, cosmo, source):
        source.scaling *= (1+ self.values['m'])   



class AdditiveShearBias(OutputSystematic):
    pass

class LinearAlignment(SourceSystematic):
    params = ['biasia']
    optional_params = {
        'alphaz': 0.0,
        'z_piv': 0.0,
        'fred': 1.0,
        'alphag': 0.0
    }
    def adjust_source(self,cosmo,source):
        pref=1.
        if self.values['alphaz']:
            pref *= ((1.+source.z)/(1.+self.values['z_piv']))**self.values['alphaz']
        if self.values['alphag']:
            pref *= ccl.growth_factor(cosmo,1./(1.+source.z))**self.values['alphag']
        
        source.ia_amplitude[:]=pref*self.values['biasia']
        source.f_red[:]=self.values['fred']

class BaryonEffects(CosmologySystematic):
    pass
