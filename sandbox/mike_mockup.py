import numpy as np

class TheoryVector(object):
    """A class that can generate a theory vector.
    
    Inputs are an input cosmology, some sources, some target statistics to include, and 
    possibly some systematics.  Most systematics are attached to the sources directly,
    but some may be passed here as more global or cosmological systematics.
    """
    def __init__(self, cosmology, sources, statistics, systematics=None):
        self.cosmology = cosmology
        self.sources = sources
        self.statistics = statistics

        # Build our complete list of systematics
        # First, any systematics given directly here are global-level systematics, not tied
        # to any particular source. E.g. baryon effects.
        if systematics is None:
            self.systematics = []
        else:
            self.systematics = systematics

        # Add in any systematics that the sources define
        for source in self.sources:
            self.systematics += source.systematics

        self.validate()

    def validate(self):
        """Validate the inputs to make sure everything in consistent.
        """
        # Check that our sources have unique names.  Maybe other sanity checks about the sources.
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("Some sources have identical names!")
        # Other sanity checks? ... 

        # Check that we have the appropriate sources to do each statistic, that the cosmology
        # is valid for this statistic, maybe other sanity checks.
        for stat in self.statistics:
            stat.validate(self.cosmology, self.sources)

    def get_params(self):
        """Build a parameter vector.
        
        These are the values that need to be chained over, including both cosmological parameters
        of interest, and nuisance parameters.

        Mostly the calling routine probably just needs the length of this vector
        """
        # Start with the variable cosmological parameters
        params = self.cosmology.get_params()

        self._param_index = [0] * (len(self.systematics)+1)
        self._param_index[0] = len(params)

        # Add in nuisance parameters
        for i_sys, sys in enumerate(self.systematics):
            sys_params = sys.get_nuisance_params(self.sources)
            params += sys_params
            self._param_index[i_sys+1] = len(params)

        return params


    def build_vector(self, params):
        """Build up a vector for a given step in a chain
        """
        # Get a new cosmology for this step in the chain.
        cosmology = self.cosmology.with_params(params[0:self._param_index[0]])

        systematics = []
        for i_sys, sys in enumerate(self.systematics):
            sys_params = params[self._param_index[i_sys]:self._param_index[i_sys+1]]
            systematics.append(sys.with_params(sys_params))

        # Presumably do any initial calculations that are needed.  Pdelta, etc.?
        cosmology.initialize()

        # Note: some systamtics might be in more than one of these catagories.
        input_sys = [ sys for sys in self.systematics if sys.affects_input() ]
        output_sys = [ sys for sys in self.systematics if sys.affects_output() ]
        calc_sys = [ sys for sys in self.systematics if sys.requires_calculation() ]

        for sys in input_sys:
            # Not exactly sure if I understood the meaning of input systematics, but maybe
            # something like this to adjust the nominal Pdelta, etc.
            cosmology = sys.adjust_input(cosmology)

        # Build up the statistcs vectors
        stat_vectors = []
        for stat in self.statistics:
            # I guess the calculation statistics need to be passed into this function.
            calc_sys = [ sys for sys in calc_sys if sys.is_relevant_to(stat) ]
            v = stat.build_vector(cosmology, self.sources, calc_sys)

            # Adjust the output vectors as needed
            for sys in output_sys:
                sys.adjust_output(v, stat, cosmology)

        # Combine into a single theory vector
        return np.concatenate(stat_vectors)
