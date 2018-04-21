from ..base_calculator import TheoryCalculator
from .theory_results import TwoPointTheoryResults

class TwoPointTheoryCalculator(TheoryCalculator):
    def __init__(self, config, metadata):
        super().__init__(config, metadata)

    def run(self, parameters):
        print("Running 2pt theory prediction")
        results = TwoPointTheoryResults(...)
        return results

