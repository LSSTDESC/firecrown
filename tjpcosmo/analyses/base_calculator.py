from ..systematics import Systematic, SourceSystematic, OutputSystematic, CosmologySystematic
from .sources import make_source


class TheoryCalculator:
    def __init__(self, config, metadata):
        self.config = config
        self.metadata = metadata
        
        """ We need to load in details of the models"""
        # config_file = yaml.load(config.open())
        
        self.source_by_name = config['name']
        self.setup_systematics(config['systematics'])
        self.setup_sources(config)

    def validate(self):
        """Validating the inputs, This function is missing for now, implement it later"""



    def setup_systematics(self, sys_config):
        self.systematics = {}
        for name, info in sys_config.items():
            sys = Systematic.from_info(name, info)
            self.systematics[name] = sys
        
        self.output_systematics = {
            name:sys
            for name,sys in self.systematics.items()
            if isinstance(sys,OutputSystematic)
        }

        self.cosmology_systematics = {
            name:sys
            for name,sys in self.systematics.items()
            if isinstance(sys,CosmologySystematic)
        }

        self.source_systematics = {
            name:sys
            for name,sys in self.systematics.items()
            if isinstance(sys,SourceSystematic)
        }




    def setup_sources(self, config):
        info = config['sources']
        used_systematics = set()
        self.sources = []

        for name, source_info in info.items():
            stype = source_info['type']

            source = make_source(name, stype, self.metadata)

            sys_names = source_info.get('systematics', [])
            # check if string or list
            for sys_name in sys_names:
                sys = self.systematics.get(sys_name)
                if sys is None:
                    raise ValueError(f"Systematic with name {sys_name} was specified for source {name} but not defined in parameter file systematics section")
                source.systematics.append(sys)
                used_systematics.add(sys_name)
            self.sources.append(source)

        for sys_name in self.source_systematics.keys():
            if sys_name not in used_systematics:
                raise ValueError(f"Systematic with name {sys_name} was specified in param file but never used")


    def update_systematics(self, parameters):
        for sys in self.systematics.values():
            sys.update(parameters)


    def apply_source_systematics(self, cosmo):
        for source in self.sources:
            source.reset()
            for syst in source.systematics:
                if isinstance(syst, SourceSystematic):
                    syst.adjust_source(cosmo, source)
