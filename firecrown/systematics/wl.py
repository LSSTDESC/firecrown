from .base_systematic import (
    SourceSystematic, OutputSystematic, CosmologySystematic)
import pyccl as ccl


class MultiplicativeShearBias(SourceSystematic):
    modified_source_properties = ['scaling']
    required_source_properties = []
    params = ['m']

    def adjust_source(self, cosmo, source):
        if self.adjust_requirements(source):
            source.scaling *= (1.0 + self.values['m'])
            return 0
        else:
            print(
                f"{self.__class__.__name__} did not find all "
                "required source parameters")
            return 1


class AdditiveShearBias(OutputSystematic):
    pass


class LinearAlignment(SourceSystematic):
    modified_source_properties = ['ia_amplitude', 'f_red']
    required_source_properties = ['z']
    params = ['biasia']
    optional_params = {
        'alphaz': 0.0,
        'z_piv': 0.0,
        'fred': 1.0,
        'alphag': 0.0
    }

    def adjust_source(self, cosmo, source):
        if self.adjust_requirements(source):
            pref = 1.0
            if self.values['alphaz']:
                pref *= (
                    ((1.0 + source.z) / (1.0 + self.values['z_piv'])) **
                    self.values['alphaz'])
            if self.values['alphag']:
                pref *= ccl.growth_factor(
                    cosmo, 1.0 / (1.0 + source.z)) ** self.values['alphag']

            source.ia_amplitude[:] = pref*self.values['biasia']
            source.f_red[:] = self.values['fred']
            return 0
        else:
            print(
                f"{self.__class__.__name__} did not find all "
                "required source parameters")
            return 1


class BaryonEffects(CosmologySystematic):
    pass
