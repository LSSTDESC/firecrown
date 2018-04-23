from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic
import pyccl as ccl


class LinearBias(SourceSystematic):
    params = ['b']
    optional_params = {
        'alphaz': 0.0,
        'z_piv': 0.0,
        'alphag': 0.0
    }
    def adjust_source(self, cosmo, source):
        pref=1.0
        if self.values['alphaz']:
            pref *= ((1.+source.z)/(1.+self.values['z_piv']))**(self.values['alphaz'])
        if self.values['alphag']:
            pref *= ccl.growth_factor(cosmo,1./(1.+source.z))**self.values['alphag']
            
        source.bias[:] = pref*self.values['b']
        #to do: we want to allow for growth dependent bias (e.g. growth*bias = const).
        # but we currently don't have the growth parameters until we run ccl.