from .dataset import TwoPointDataSet
from .calculator import TwoPointTheoryCalculator
from .theory_results import TwoPointTheoryResults
from ..base_analysis import Analysis

class TwoPointAnalysisModel(Analysis):
    name = '_'
    theory_calculator_class = TwoPointTheoryCalculator
    data_class = TwoPointDataSet
    metadata_class = None

    def extract_metadata(self, data_info):
        return None

