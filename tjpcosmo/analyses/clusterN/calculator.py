import pyccl as ccl
from ..base_calculator import TheoryCalculator


class ClusterNCalculator(TheoryCalculator):
    def __init__(self, config, metadata):
        super().__init__(config, metadata)


    def apply_output_systematics(self, c_ell_pair):
        pass


    def run(self, results, parameters):
        print("Running Cluster N theory prediction")
        self.update_systematics(parameters)
        params = convert_cosmobase_to_ccl(parameters)
        cosmo=ccl.Cosmology(params)
        self.apply_source_systematics(cosmo)

        # Call this somehow
        # results.add('twopoint', src1, src2, ells, c_ell_pair)

