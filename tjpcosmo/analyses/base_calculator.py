class TheoryCalculator:
    def __init__(self, config, metadata):
        self.config = config
        self.metadata = metadata
        
        """ We need to load in details of the models"""
        # config_file = yaml.load(config.open())
        
        self.source_by_name = config['name']
    
        
    def validate(self):
        """Validating the inputs, This function is missing for now, implement it later"""
