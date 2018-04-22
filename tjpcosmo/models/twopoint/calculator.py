from ..base_calculator import TheoryCalculator
from .theory_results import TwoPointTheoryResults
from ...systematics import Systematic, SourceSystematic, OutputSystematic, CosmologySystematic
from ..sources import make_source, LSSSource

def make_fake_source(name, stype, metadata):
    import numpy as np

    z = np.arange(0.0, 2.0, 0.01)
    n_of_z = np.exp(-0.5 * (z - 0.8)**2/0.1**2)
    metadata = {name:{"nz":(z,n_of_z)}}
    return LSSSource(name, stype, metadata)


class TwoPointTheoryCalculator(TheoryCalculator):
    def __init__(self, config, metadata):
        super().__init__(config, metadata)
        print("Warning! Making a fake source")
        self.setup_systematics(config['systematics'])
        self.sources = self.make_sources(config)
        self.metadata = metadata

    def setup_systematics(self, sys_config):
        self.systematics = {}
        for name, info in sys_config.items():
            sys = Systematic.from_info(info)
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


    def make_sources(self, config):
        info = config['sources']
        used_systematics = set()

        for source_info in info:
            stype = source_info['type']
            sname = source_info['name']

            source = make_fake_source(sname, stype, self.metadata)

            sys_names = source_info.get('systematics', [])
            # check if string or list
            for sys_name in sys_names:
                sys = self.systematics.get(sys_name)
                if sys is None:
                    raise ValueError(f"Systematic with name {sys_name} was specified for source {sname} but not defined in parameter file systematics section")
                source.systematics.append(sys)
                used_systematics.add(sys_name)

        for sys_name in self.source_systematics.keys():
            if sys_name not in used_systematics:
                raise ValueError(f"Systematic with name {sys_name} was specified in param file but never used")

    def make_tracers(self):
        pass

    def run(self, parameters):
        print("Running 2pt theory prediction")

        self.update_systematics(parameters)
        cosmo = CCL.cosmo(parameters)

        tracers = self.make_tracers()

        for twopoint_pair in something:
            tracer1 = ccl.CLTracer(..., tracer_type=...)
            tracer2 = ccl.CLTracer(...)
            CCL.XXX(cosmo, tracer, tracer2)

        results = TwoPointTheoryResults(...)
        return results

