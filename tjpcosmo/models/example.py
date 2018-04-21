from .base_analysis import Analysis
from .base_calculator import TheoryCalculator
from .base_theory_results import TheoryResults
from ..dataset import BaseDataSet
import yaml
import numpy as np

class ExampleDataSet(BaseDataSet):
    def __init__(self, mu, sigma):
        self.data_vector = np.array([mu])
        self.covariance = np.array([[sigma**2]])
        self.precision = np.array([[sigma**-2]])

    @classmethod
    def load(cls, data_info):
        print('info = ', data_info)
        ombh2_info = data_info['bbn']
        ombh2_file = ombh2_info['file']
        ombh2_data = yaml.load(open(ombh2_file))
        mu = ombh2_data['mean']
        sigma = ombh2_data['sigma']
        return cls(mu, sigma), None



class ExampleTheoryResults(TheoryResults):
    slots = ['ombh2']
    def data_vector(self):
        return np.array([self.ombh2])


class ExampleTheoryCalculator(TheoryCalculator):
    def __init__(self, config, metadata):
        super().__init__(config, metadata)

    def run(self, parameters):
        print("Running Example theory prediction")
        omega_b = parameters.Omega_b
        h = parameters.h

        results = ExampleTheoryResults()
        results.ombh2 = omega_b * h**2
        return results

class ExampleAnalysisModel(Analysis):
    name = 'example'
    theory_calculator_class = ExampleTheoryCalculator
    theory_results_class = ExampleTheoryResults
    data_class = ExampleDataSet
    metadata_class = None

    def extract_metadata(self, data_info):
        return None
