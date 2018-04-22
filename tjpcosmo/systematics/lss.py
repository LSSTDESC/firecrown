from .base_systematic import SourceSystematic, OutputSystematic, CosmologySystematic


class LinearBias(SourceSystematic):
    params = ['b']
    def adjust_source(self, cosmo, source):
        source.bias[:] = self.values['b']