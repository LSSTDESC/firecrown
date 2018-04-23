from .dataset import TwoPointDataSet
from .calculator import TwoPointTheoryCalculator
from .theory_results import TwoPointTheoryResults
from ..base_analysis import Analysis

class TwoPointAnalysis(Analysis):
    name = 'twopoint'
    theory_calculator_classes = [TwoPointTheoryCalculator]
    theory_results_class = TwoPointTheoryResults
    data_class = TwoPointDataSet
    metadata_class = None

    def extract_metadata(self, data_info):
        return None


# class TwoPointClusterNAnalysis(Analysis):
#     name = 'twopoint'
#     theory_calculator_classes = [TwoPointTheoryCalculator, ClusterNTheoryCalculator]
#     data_class = TwoPointClusterNDataSet
#     metadata_class = None

#     def extract_metadata(self, data_info):
#         return None

