from ..base_theory_results import TheoryResults
import numpy as np

class TwoPointTheoryResults(TheoryResults):
    slots = ['lots_of_stuff?']
    def __init__(self, c_ell_info):
        vector = []
        self.sources = []
        for info in c_ell_info:
            (src1,src2,ells,c_ell) = info
            self.sources.append((src1,src2))
            vector.append(c_ell)
        self.vector = np.concatenate(vector)

    def to_cosmosis_block(self, block):
        pass

    def data_vector(self):
        # order the data in the same order as the DataSet
        return self.vector

