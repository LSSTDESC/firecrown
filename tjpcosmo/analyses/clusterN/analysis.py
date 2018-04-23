from .calculator import ClusterNCalculator
from .theory_results import ClusterNTheoryResults
from .dataset import ClusterNDataSet
from ..base_analysis import Analysis

class ClusterNAnalysis(Analysis):
    name = 'clusterN'
    theory_calculator_classes = [ClusterNCalculator]
    theory_results_class = ClusterNTheoryResults
    data_class = ClusterNDataSet
    metadata_class = None

    def extract_metadata(self, data_info):
        return None

