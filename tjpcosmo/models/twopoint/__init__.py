from .dataset import TwoPointDataSet
from .calculator import TwoPointTheoryCalculator
from .theory_results import TwoPointTheoryResults
from ..base_model import AnalysisModel

class TwoPointAnalysisModel(AnalysisModel):
    name = '_'
    theory_calculator_class = TwoPointTheoryCalculator
    theory_results_class = TwoPointTheoryResults
    data_class = TwoPointDataSet
    metadata_class = None

    def extract_metadata(self, data_info):
        return None
