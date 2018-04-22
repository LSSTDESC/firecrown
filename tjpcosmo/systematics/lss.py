from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic


class LinearBias(SourceSystematic):
    params = ['b']
    optional_params = {
        'alpha': 0.0,
        'z_piv':0.0
    }
    def adjust_source(self, cosmo, source):
        source.bias[:] *= ((1.+source.z)/(1.+self.values['z_piv']))**(self.values['alpha'])*self.values['b']