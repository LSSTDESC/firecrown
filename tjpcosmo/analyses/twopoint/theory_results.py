from ..base_theory_results import TheoryResults
import numpy as np

class TwoPointTheoryResults(TheoryResults):
    slots = ['lots_of_stuff?']
    def __init__(self, metadata):
        self.vector = []
        self.sources = []

    def add(self, data_type, src1, src2, ells, c_ell_pair):
        self.sources.append((src1,src2))
        self.vector.append(c_ell_pair)

    def to_cosmosis_block(self, block):
        pass

    def data_vector(self):
        # order the data in the same order as the DataSet
        return np.concatenate(self.vector)

