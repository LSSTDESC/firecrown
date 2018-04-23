model_registry = {}


class Analysis:
    """
    This is the superclass for all analyses, which represent the part of the
    likelihood function that generate mean theory predictions from parameters.

    This is just the skeleton - there are various more complete designs in the sandbox.
    Delete this notice after implementing them!

    """
    theory_calculator_class = None
    theory_results_class = None
    data_class = None
    metadata_class = None


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.name if hasattr(cls, 'name') else cls.__name__
        name = name.lower()
        print(f"Register {name}")
        model_registry[name] = cls




    def __init__(self, config, data_info, likelihood_class):
        """
        Instantiate the model from a dictionary of options.

        Subclasses usually override this to do their own instantiation.
        They should call this parent method first.
        """
        self.config=config
        self.data, self.metadata = self.data_class.load(data_info, config)
        
        self.theory_calculators = [
            cls(config, self.metadata)
            for cls in self.theory_calculator_classes]

        self.likelihood = likelihood_class(self.data)

    @staticmethod
    def from_name(name):
        return model_registry[name]


    def run(self, parameterSet):
        theory_results = self.theory_results_class(self.metadata)
        for calculator in self.theory_calculators:
             calculator.run(theory_results, parameterSet)
        like = self.likelihood.run(theory_results)
        return like, theory_results

    def likelihood(self, parameterSet):
        return self.run[0]



