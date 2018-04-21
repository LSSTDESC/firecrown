from .base_systematics import SourceSystematic, OutputSystematic


class MultiplicativeShearBias(SourceSystematic):
    pass

class PZTransformation(SourceSystematic):
    pass

class PZShift(PZTransformation):
    pass

class AdditiveShearBias(OutputSystematic):
    pass