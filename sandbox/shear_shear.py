class ShearShearDataVector(DataVector):
    def __init__(self, *args, **kwargs):
        additive_bias = kwargs['additive_bias']
        atmospheric_power_model = kwargs['atmospheric_power_model']
        
        systematics_model_list = []
        systematics_model_list.append(additive_bias)
        systematics_model_list.append(atmospheric_power_model)

        self.systematics_mod
        self.source_galaxies = kwargs['source_galaxies']
        
        for model in kwargs['models']:
            try:
                for systematics_model in model.systematics_models:
                    systematics_model_list.append(systematics_model.model_name)
                    
            except AttributeError:
                pass

        DataVector.__init__(self, systematics_models=systematics_model_list)