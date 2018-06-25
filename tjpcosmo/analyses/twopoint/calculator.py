import pyccl as ccl
from ..base_calculator import TheoryCalculator
from ..base_theory_results import TheoryResults
import numpy as np



def ell_for_xi() :
    """ Returns an array of ells that should hopefully sample the
    power spectrum sufficiently well
    """
    #TODO: these values are hard-coded right now
    ell_min=2
    ell_mid=50
    ell_max=6E4
    return np.concatenate((np.linspace(ell_min,ell_mid-1,ell_mid-ell_min),
                           np.logspace(np.log10(ell_mid),np.log10(ell_max),200)))
        


class TwoPointTheoryCalculator(TheoryCalculator):
    # This needs to be added by hand for every TheoryCalculator subclass
    name = 'twopoint'
    statistic_types = ['ClGG', 'ClGE', 'ClEE', 'XiGG', 'XiGE', 'XiP', 'XiM']

    def apply_output_systematics(self, c_ell_pair):
        pass

    def make_tracers(self, cosmo):
        tracers = {}
        for source in self.sources:
            tracers[source.name] = (source.to_tracer(cosmo), source.scaling)
        return tracers

    def run(self, cosmo, parameters, results):
        print("Running 2pt theory prediction")
        tracers = self.make_tracers(cosmo)

        dict_xcor_ccl={'XiGG':'gg','XiGE':'gl','XiP':'l+','XiM':'l-'}
        for pair_info in self.metadata['ordering']:
            src1 = pair_info['src1']
            tracer1, scaling1 = tracers[src1]
            src2 = pair_info['src2']
            tracer2, scaling2 = tracers[src2]
            xcor_type=pair_info['type']
            xs = pair_info['xs']
            stat_name = pair_info['name']
            scaling = scaling1 * scaling2

            if xcor_type.startswith('Cl'):
                xcor_pair = ccl.angular_cl(cosmo, tracer1, tracer2, xs)
            elif xcor_type.startswith('Xi'): #Configuration space
                larr=ell_for_xi()
                clarr=ccl.angular_cl(cosmo, tracer1, tracer2, larr)
                xcor_pair=ccl.correlation(cosmo,larr,clarr,xs/60,
                                          corr_type=dict_xcor_ccl[xcor_type])
            else:
                raise ValueError("Unknown cross-correlation type for this code")

            xcor_pair*=scaling
            self.apply_output_systematics(xcor_pair)
            results.set(stat_name, xcor_pair)
