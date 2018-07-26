from ..systematics import Systematic, SourceSystematic, OutputSystematic, CosmologySystematic
from ..sources import make_source
import collections

#
calculator_registry = collections.defaultdict(list)


class TheoryCalculator:
    def __init__(self, config, metadata, sources, systematics):
        self.config = config
        self.metadata = metadata
        self.systematics = systematics
        self.sources = sources
        
        """ We need to load in details of the models"""
        # config_file = yaml.load(config.open())
        

    def validate(self):
        """Validating the inputs, This function is missing for now, implement it later"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.name if hasattr(cls, 'name') else cls.__name__
        name = name.lower()
        statistics_calculator_can_compute = getattr(cls, 'statistic_types', [])

        for s in statistics_calculator_can_compute:
            calculator_registry[s].append(cls)

    @classmethod
    def calculator_for_statistic(cls, stat, choice=None):
        calcs = calculator_registry[stat]

        if len(calcs)==0:
            raise ValueError(f"No known calculator can compute statistic: {stat}")

        if choice is None:
            if len(calcs)==1:
                calc = calcs[0]
            else:
                raise ValueError(f"Multiple calculators ({calcs}) can compute statistics '{stat}'. Please set 'choice' option in it to pick")
        else:
            chosen_calcs = [c for c in calcs if c.name==choice]
            if len(chosen_calcs)==1:
                calc = chosen_calcs[0]
            elif len(chosen_calcs)==0:
                raise ValueError(f"Your chosen calculator '{calc}' does not know how to compute statistic '{stat}'")
            else:
                raise ValueError(f"Multiple calculators ({chosen_calcs}) have the same name {choice}!  This is not allowed")
        return calc




