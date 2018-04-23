from ..base_theory_results import TheoryResults
import numpy as np

class ClusterNTheoryResults(TheoryResults):
    slots = []
    def __init__(self, metadata):
        self.vector = []
        self.other_useful_info = []

    def add(self, arg1, arg2, theory_chunk, **more_args):
        self.vector.append(theory_chunk)

    def to_cosmosis_block(self, block):
        pass

    def data_vector(self):
        # order the data in the same order as the DataSet
        return np.concatenate(self.vector)

