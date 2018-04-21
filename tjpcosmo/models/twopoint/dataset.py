from ...dataset import BaseDataSet

class TwoPointDataSet(BaseDataSet):
    def __init__(self, sacc_data):
        self.data_vector = ...
        self.covariance = ...
        self.precision = ...

    @classmethod
    def load(cls, data_info):
        sacc_data = ...
        return cls(sacc_data)


