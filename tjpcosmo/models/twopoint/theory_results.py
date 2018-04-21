from ..base_theory_results import TheoryResults


class TwoPointTheoryResults(TheoryResults):
    slots = ['lots_of_stuff?']
    def data_vector(self):
        # order the data in the same order as the DataSet
        return np.array([...])

