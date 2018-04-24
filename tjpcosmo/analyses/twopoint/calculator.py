import pyccl as ccl
from ..base_calculator import TheoryCalculator
import numpy as np
def make_fake_source(name, stype, metadata):
    import numpy as np

    z = np.arange(0.0, 2.0, 0.01)
    n_of_z = np.exp(-0.5 * (z - 0.8)**2/0.1**2)
    metadata = {
        "sources":{
            name: {
                "nz": [z, n_of_z]
            }
        }
    }
    return LSSSource(name, stype, metadata)

        # self.z,self.nz = metadata['sources'][name]["nz"]

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
        
def convert_cosmobase_to_ccl(cosmo_base):
    """ Function for changing our set of parameters from the DESC standard set
    forth by Phil Bull, to a format that can be put into CCL.
    """
    # Although these are lower case they are fractions
    # not densities - this is an artifact of cosmosis
    # case insesitivity
    omega_c = cosmo_base.omega_c
    omega_b = cosmo_base.omega_b
    omega_k = cosmo_base.omega_k
    if (cosmo_base.omega_n_rel + cosmo_base.omega_n_mass):
        raise ValueError("cosmo_base doesn't handle massive neutrinos yet")
    w = cosmo_base.w0
    wa = cosmo_base.wa
    h0 = cosmo_base.h
    if 'sigma_8' in cosmo_base:
        sigma8 = cosmo_base.sigma_8
    else:
        sigma8 = 0.0

    if 'a_s' in cosmo_base:
        A_s = cosmo_base.a_s
    else:
        A_s = 0.0

    n_s = cosmo_base.n_s

    print("Baryon systematics go into convert_cosmobase_to_ccl")


    if sigma8 and A_s:
        raise ValueError("Specifying both sigma8 and A_s: pick one")
    elif sigma8:
        params=ccl.Parameters(Omega_c=omega_c,Omega_b=omega_b,Omega_k=omega_k,
                              w0=w,wa=wa,sigma8=sigma8,n_s=n_s,h=h0)
    elif A_s:
        params = ccl.Parameters(Omega_c=omega_c,Omega_b=omega_b,Omega_k=omega_k,
                                w0=w,wa=wa,A_s=A_s,n_s=n_s,h=h0)
    else:
        raise ValueError("Need either sigma 8 or A_s in pyccl.")


    return params

class TwoPointTheoryCalculator(TheoryCalculator):
    def __init__(self, config, metadata):
        super().__init__(config, metadata)


    def apply_output_systematics(self, c_ell_pair):
        pass

    def make_tracers(self, cosmo):
        tracers = {}
        for source in self.sources:
            tracers[source.name] = (source.to_tracer(cosmo), source.scaling)
        return tracers

    def run(self, results, parameters):
        print("Running 2pt theory prediction")
        print(parameters)
        
        self.update_systematics(parameters)

        params = convert_cosmobase_to_ccl(parameters)
        print("Calling CCL with default config - may need to change depending on systematics/choices")
        cosmo=ccl.Cosmology(params)
                            #transfer_function=dic_par['transfer_function'],
                            #matter_power_spectrum=dic_par['matter_power_spectrum'])
        self.apply_source_systematics(cosmo)

        tracers = self.make_tracers(cosmo)

        dict_xcor_types={'ClGG':100,'ClGE':100,'ClEE':100,
                         'XiGG':200,'XiGE':201,'XiP':202,'XiM':203}
        dict_xcor_ccl={'XiGG':'gg','XiGE':'gl','XiP':'l+','XiM':'l-'}
        for pair_info in self.metadata['2pt_ordering']:
            src1 = pair_info['src1']
            tracer1, scaling1 = tracers[src1]
            src2 = pair_info['src2']
            tracer2, scaling2 = tracers[src2]
            xcor_type=pair_info['type']
            xs = pair_info['xs']
            scaling = scaling1 * scaling2

            if dict_xcor_types[xcor_type]==dict_xcor_types['ClGG'] : #Fourier space
                xcor_pair = ccl.angular_cl(cosmo, tracer1, tracer2, xs)
            else : #Configuration space
                larr=ell_for_xi()
                clarr=ccl.angular_cl(cosmo, tracer1, tracer2, larr)
                xcor_pair=ccl.correlation(cosmo,larr,clarr,xs/60,
                                          corr_type=dict_xcor_ccl[xcor_type])

            xcor_pair*=scaling
            self.apply_output_systematics(xcor_pair)
            results.add('twopoint', src1, src2, xs, xcor_pair)
