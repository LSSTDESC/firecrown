from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic
import pyccl as ccl


class LinearBias(SourceSystematic):
    modified_source_properties =['bias']
    required_source_properties =['z']
    params = ['b']
    optional_params = {
        'alphaz': 0.0,
        'z_piv': 0.0,
        'alphag': 0.0
    }
    def adjust_source(self, cosmo, source):
        if self.adjust_requirements(source):
            pref=1.0
            if self.values['alphaz']:
                pref *= ((1.+source.z)/(1.+self.values['z_piv']))**(self.values['alphaz'])
            if self.values['alphag']:
                pref *= ccl.growth_factor(cosmo,1./(1.+source.z))**self.values['alphag']
            
            source.bias[:] = pref*self.values['b']
            return 0
        else:
            print(f"{self.__class__.__name__} did not find all required source parameters")
            return 1
