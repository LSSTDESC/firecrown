systematic_registry = {}
import copy

class Systematic:
    """ Super class for systematics, we have assumed there are 3 types of places
    systematics can live, either in the source, cosmology or in the output.
    
    4/23 Only source systematics is actually implemented. 
    """
    params = []
    optional_params = {}
    
    def __init__(self, name, **config):
        self.name = name
        self.config = config
        self.values = {}
        print(f"Would now create systematic {self.__class__.__name__} from config info: {config}")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.name if hasattr(cls, 'name') else cls.__name__
        name = name.lower()
        print(f"Register systematic {name}")
        systematic_registry[name] = cls

    @classmethod
    def from_info(cls, name, config):
        config  = config.copy()
        class_name = config.pop('type')
        if class_name is None:
            raise ValueError("Systematic is missing 'type' entry in param file")
        class_obj = systematic_registry.get(class_name.lower())
        if class_obj is None:
            raise ValueError(f"Systematic called {class_name} not known")

        systematic = class_obj(name, **config)
        return systematic

    def adjust_source(self, cosmo, source):
        print(f"Systematics {self.name} is NOT IMPLEMENTED!")

    def update(self, parameters):
        for param in self.params:
            v = parameters[f"{self.name}.{param}"]
            self.values[param] = v

        for param, default in self.optional_params.items():
            v = parameters.get(f"{self.name}.{param}")
            if v is None:
                v = default
            self.values[param] = v


class CosmologySystematic(Systematic):
    pass

class SourceSystematic(Systematic):
    modified_source_properties = []
    required_source_properties = []
    def adjust_requirements(self,source):
        if (len(self.modified_source_properties)==0):
        #move this check to init?
            print(f"Systematic {self.__class__.__name__} does not modify any source properties!")
            print(f"If this is unintended, register them in {self.__class__.__name__}.modified_source_properties")
            
        #print(f"required {self.required_source_properties}")
        #print(f"existing {source.eval_source_prop}")
        if set(self.required_source_properties) <= set(source.eval_source_prop):
            if (len(self.modified_source_properties)):
                source.eval_source_prop.extend(self.modified_source_properties)
                #print(f"extended {source.eval_source_prop}\n")
            return True
        else:
            return False

class OutputSystematic(Systematic):
    pass