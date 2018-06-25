from .twopoint import TwoPointDataSet
from ..sources import make_source
from .base_calculator import TheoryCalculator
from .base_theory_results import TheoryResults
from ..systematics import Systematic, OutputSystematic, CosmologySystematic, SourceSystematic
from ..likelihood import BaseLikelihood
import collections
import pyccl as ccl

class Analysis:
    """
    An Analysis object collects together the various pieces needed to calculate
    a likelihood of a collected group of correlated probes.

    It incorporates the theory_calculators that compute the theory predictions,
    systematic error models and the sources to which they apply, the observed data,
    and the likelihood functional form that compares the theory to the data.

    A new Analysis object is created by the Analysis.from_dict class method.

    The Analysis.run is the main method called from outside the class.

    
    """


    def __init__(self, name, theory_calculators, systematics, sources, likelihood, data, metadata):
        """
        Create an Analysis object by specifying all its ingredients.
        """
        self.name = name
        self.theory_calculators = theory_calculators
        self.sources = sources
        self.likelihood = likelihood
        self.data = data
        self.metadata = metadata
        self.systematics = systematics

        self.classify_systematics()




    @classmethod
    def from_dict(cls, name, info):
        """
        Build an Analysis object from a dictionary describing its various
        ingredients.
        """

        # Call other class methods that create the different components of an Analysis
        data, metadata = cls.create_data(info)
        systematics = cls.create_systematics(info)
        sources = cls.create_sources(info, systematics, metadata)
        calculators = cls.create_calculators(info, systematics, sources, metadata)
        likelihood = cls.create_likelihood(info, data)

        return cls(name, calculators, systematics, sources, likelihood, data, metadata)



    @classmethod
    def create_data(cls, info):
        """

        """
        filename = info['data']['filename']
        if filename.endswith(".sacc"):
            data = TwoPointDataSet.load(filename, info)
        else:
            raise ValueError("We have not written code to load anything except SACC 2pt files yet.")
        return data



    @classmethod
    def create_sources(cls, info, systematics, metadata):
        sources_info = info['sources']

        sources = []

        for name, source_info in sources_info.items():
            stype = source_info['type']

            source = make_source(name, stype, metadata)
            print(f"Made source '{name}' of type {stype}")

            sys_names = source_info.get('systematics', [])
            # check if string or list
            for sys_name in sys_names:
                sys = systematics.get(sys_name)
                if sys is None:
                    raise ValueError(f"Systematic with name {sys_name} was specified for source {name} but not defined in parameter file systematics section")
                source.systematics.append(sys)
                print(f"    Attaching systematics model {sys_name} to source '{name}'")

            sources.append(source)

        return sources

    @classmethod
    def create_systematics(cls, info):
        systematics_info = info['systematics']

        systematics = {}
        for name, sys_info in systematics_info.items():
            sys = Systematic.from_info(name, sys_info)
            systematics[name] = sys

        return systematics


 

    @classmethod
    def create_calculators(cls, info, systematics, sources, metadata):
        stats_info = info['statistics']

        source_dict = {s.name:s for s in sources}

        calc_types = set()

        output_sys = collections.defaultdict(dict)
        sources_for_calcs = collections.defaultdict(set)

        for name, stat_info in stats_info.items():
            stat_type = stat_info['type']
            choice = stat_info.get('calculator', None)

            # Find a theory class that can calculate this statistic
            calc_type = TheoryCalculator.calculator_for_statistic(stat_type, choice)
            calc_types.add(calc_type)

            # Get the systematics that apply to this statistic
            # and record them so that the class can apply them
            sys_names = stat_info.get('systematics', [])
            output_sys[calc_type][name] = [systematics.get(sys_name) for sys_name in sys_names]

            sources_for_stat = stat_info['source_names']
            for s in sources_for_stat:
                sources_for_calcs[calc_type].add(source_dict[s])

        calcs = [calc_type(info, metadata, sources_for_calcs[calc_type], output_sys[calc_type]) for calc_type in calc_types]

        return calcs

    @classmethod
    def create_likelihood(cls, info, data):
        like_info = info['likelihood']
        like_type = like_info['type']
        like_class = BaseLikelihood.from_name(like_type)
        # TODO Make likelihood accept options here
        like = like_class(data)
        return like


    def update_systematics(self, parameters):
        for sys in self.source_systematics.values():
            sys.update(parameters)
        for sys in self.output_systematics.values():
            sys.update(parameters)



    def apply_source_systematics(self, cosmo):
        for source in self.sources:
            source.apply_systematics(cosmo)

    def apply_output_systematics(self):
        for sys in self.output_systematics.values():
            print("Need to write code to apply output systematics!")


    def run(self, cosmo, parameterSet):
        """
        Run the analysis on the given cosmology and parameterSet.

        Parameters
        ----------

        cosmo: CCL.Cosmology object
            The (pre-initialized) cosmology

        parameterSet: ParameterSet object
            Container of other parameters

        Returns
        -------

        like: float
            Likelihood value

        theory_results: TheoryResults object
            Collection of theory predictions for the measured values

        """

        # Update all the systematics that apply to this
        # analysis
        self.update_systematics(parameterSet)

        # Apply any systematics that modify the sources
        self.apply_source_systematics(cosmo)

        # Create the structure which will hold the different theory prediction numbers
        theory_results = TheoryResults(self.metadata)

        # Run each of our calculators in turn
        # Each will fill in a different bit of the theory_results object
        for calculator in self.theory_calculators:
            calculator.run(cosmo, parameterSet, theory_results)

        # Apply any systematics that modify the theory_results at the end
        self.apply_output_systematics()

        # Compute the likelihood value
        like = self.likelihood.run(theory_results)

        # Return both the likelihood number and the theory results object
        return like, theory_results



    def classify_systematics(self):
        """
        Split up the Systematics that apply to this analysis into
        output and source type systematics.
        """
        self.output_systematics = {
            name:sys
            for name,sys in self.systematics.items()
            if isinstance(sys,OutputSystematic)
        }

        self.source_systematics = {
            name:sys
            for name,sys in self.systematics.items()
            if isinstance(sys,SourceSystematic)
        }



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