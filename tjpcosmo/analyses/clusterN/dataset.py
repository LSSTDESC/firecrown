from ...dataset import BaseDataSet


class ClusterNDataSet(BaseDataSet):
    def __init__(self, data, config) :
        self.data_vector = data

    @classmethod
    def load(cls, info, config):
        print("Warning: FAKING CLUSTER DATA!")

        z_bins = [
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0)
        ]

        lambda_bins = [
            (20., 30.),
            (30., 50.),
            (50., 100.),
        ]

        data = []
        metadata = []
        for (zmin, zmax) in z_bins:
            for (lambda_min, lambda_max) in lambda_bins:
                datum = (zmin, zmax, lambda_min, lambda_max, 1)
                data.append(datum)

        return data, metadata
