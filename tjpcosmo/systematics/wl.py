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
    pass

class BaryonEffects(CosmologySystematic):
    pass